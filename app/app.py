
# Import required libraries
import joblib
import numpy as np
import pandas as pd
import os
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from sklearn.model_selection import KFold

# Custom classes from the notebook to support A2 model loading
class NoRegularization:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return np.zeros_like(theta)

# Lasso Regularization (L1)
class LassoPenalty:
    def __init__(self, l):
        self.l = l # lambda value
    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))
    def derivation(self, theta):
        return self.l * np.sign(theta)

# Ridge Regularization (L2)
class RidgePenalty:
    def __init__(self, l):
        self.l = l # lambda value
    def __call__(self, theta):
        return self.l * np.sum(theta**2)
    def derivation(self, theta):
        return 2 * self.l * theta

# Custom LinearRegression class from notebook
class LinearRegression(object):
    # In this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)

    def __init__(self, 
                 regularization = None, 
                 lr=0.001, 
                 method='batch', 
                 num_epochs=500, 
                 batch_size=50, 
                 cv=kfold, 
                 init_method='zeros', 
                 use_momentum=False, 
                 momentum=0.9,
                 poly_degree=1):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization if regularization is not None else NoRegularization()

        # Addition of parameters
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.poly_degree = poly_degree

        # Check if momentum is used, and momentum value is between 0 and 1
        if self.use_momentum and not (0 < self.momentum < 1):
            raise ValueError("Momentum value must be between 0 and 1.")

    def mse(self, ytrue, ypred):
        # Verify if it is scalar or array
        if np.isscalar(ytrue):
            return (ypred - ytrue) ** 2
        else:
            return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # Add a function to compute R-squared score
    def r2(self, ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum() # Residual sum of squares
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum() # Total sum of squares
        if ss_tot == 0:
            return 0  # Avoid division by zero; return 0 if variance is zero
        return 1 - (ss_res/ss_tot)

    def poly_features(self, X, degree):
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X_train, y_train):
        # Create a list of kfold scores
        self.kfold_scores = list()

        # kfold.split in the sklearn.....
        # 3 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            # Reset val loss (Move it inside the fold loop due to early stopping)
            self.val_loss_old = np.inf
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            # Add polynomial features if degree > 1
            if self.poly_degree > 1:
                X_cross_train = self.poly_features(X_cross_train, self.poly_degree)
                X_cross_val = self.poly_features(X_cross_val, self.poly_degree)

            # Find the number of features in the dataset
            n_features = X_cross_train.shape[1]
            
            # Add xavier weights initialization
            if self.init_method == 'xavier':
                limit = np.sqrt(1.0/n_features)
                lower,upper = -limit, limit
                number = np.random.rand(n_features)
                self.theta = lower + number * (upper - lower)

            else: # Set default initialization to zeros
                self.theta = np.zeros(n_features)  

            # init once per fold
            self.prev_step = np.zeros_like(self.theta)

            # Define X_cross_train as only a subset of the data
            # How big is this subset?  => mini-batch size ==> 50

            # One epoch will exhaust the WHOLE training set
            # Note: Removed mlflow logging for app compatibility
            for epoch in range(self.num_epochs):
            
                # With replacement or no replacement
                # with replacement means just randomize
                # with no replacement means 0:50, 51:100, 101:150, ......300:323
                # Shuffle your index
                perm = np.random.permutation(X_cross_train.shape[0])
                        
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]
                
                if self.method == 'sto':
                    for batch_idx in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[batch_idx].reshape(1, -1) # (11,) ==> (1, 11) ==> (m, n)
                        y_method_train = y_cross_train[batch_idx] 
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                        # batch_idx = 0, 50, 100, 150
                        X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                        train_loss = self._train(X_method_train, y_method_train)
                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)
                val_r2_new = self.r2(y_cross_val, yhat_val)

                # Early stopping
                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new
        
            self.kfold_scores.append(val_loss_new)
            print(f"Fold {fold}: {val_loss_new}")
        
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)

        # Update with momentum if enabled
        if self.use_momentum:
            step = self.lr * grad
            update = self.momentum * self.prev_step - step
            self.theta += update
            self.prev_step = step # Store the current step for the next iteration
        else: # Standard gradient descent update
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)

    def predict(self, X, is_polynomial=False):
        if is_polynomial:
            X = self.poly_features(X, self.poly_degree)
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  # Remind that theta is (w0, w1, w2, w3, w4.....wn)
                               # w0 is the bias or the intercept
                               # w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]

# Define Lasso classes inheriting from LinearRegression
class Lasso(LinearRegression):
    def __init__(self, l=0.01, **kwargs):
        super().__init__(regularization=LassoPenalty(l), **kwargs)

# Define Ridge classes inheriting from LinearRegression
class Ridge(LinearRegression):
    def __init__(self, l=0.01, **kwargs):
        super().__init__(regularization=RidgePenalty(l), **kwargs)

class Polynomial(LinearRegression):
    def __init__(self, degree=2, **kwargs):
        super().__init__(poly_degree=degree, **kwargs)

# ManualPreprocessor class from the notebook
class ManualPreprocessor:
    def __init__(self, num_med_cols, num_mean_cols, cat_cols, drop_first=True):
        self.num_med_cols = list(num_med_cols)
        self.num_mean_cols = list(num_mean_cols)
        self.cat_cols = list(cat_cols)
        self.drop_first = drop_first
        # learned params
        self.medians_ = {}
        self.means_ = {}
        self.num_mean_for_scale_ = {}
        self.num_std_for_scale_ = {}
        self.cat_categories_ = {}
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame):
        X = X.copy()

        # 1) impute stats
        for c in self.num_med_cols:
            if c in X.columns:
                self.medians_[c] = X[c].median()
        for c in self.num_mean_cols:
            if c in X.columns:
                self.means_[c] = X[c].mean()

        # 2) impute to compute scaler on train
        for c in self.num_med_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self.medians_[c])
        for c in self.num_mean_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self.means_[c])

        # 3) scaler stats (column-wise)
        num_all = self.num_med_cols + self.num_mean_cols
        for c in num_all:
            if c in X.columns:
                self.num_mean_for_scale_[c] = X[c].mean()
                self.num_std_for_scale_[c] = X[c].std(ddof=0)
                # Ensure std is not zero
                if self.num_std_for_scale_[c] == 0:
                    self.num_std_for_scale_[c] = 1.0

        # 4) categorical categories (store train cats; unknowns will be ignored)
        for c in self.cat_cols:
            if c in X.columns:
                cats = pd.Index(pd.Series(X[c], dtype="object").dropna().unique())
                # Use a deterministic order:
                self.cat_categories_[c] = pd.Index(sorted(cats.astype(str)))

        # 5) build feature names (without bias)
        self._build_feature_names()
        self.is_fitted_ = True
        return self

    def _build_feature_names(self):
        """Helper method to build feature names"""
        num_names = self.num_med_cols + self.num_mean_cols
        cat_names = []
        for c in self.cat_cols:
            if c in self.cat_categories_:
                cats = self.cat_categories_[c]
                # drop_first=True -> drop the first category
                cats_keep = cats[1:] if self.drop_first and len(cats) > 0 else cats
                cat_names += [f"{c}={val}" for val in cats_keep]
        self.feature_names_ = np.array(num_names + cat_names, dtype=object)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()

        # 1) impute using train stats
        for c in self.num_med_cols:
            if c in X.columns and c in self.medians_:
                X[c] = X[c].fillna(self.medians_[c])
        for c in self.num_mean_cols:
            if c in X.columns and c in self.means_:
                X[c] = X[c].fillna(self.means_[c])

        # 2) scale numeric
        num_all = self.num_med_cols + self.num_mean_cols
        X_num = []
        for c in num_all:
            if c in X.columns and c in self.num_mean_for_scale_:
                mu = self.num_mean_for_scale_[c]
                sd = self.num_std_for_scale_[c]
                X_num.append(((X[c].astype(float) - mu) / sd).to_numpy())
        X_num = np.column_stack(X_num) if X_num else np.empty((len(X), 0))

        # 3) one-hot categorical using TRAIN categories
        X_cat_parts = []
        for c in self.cat_cols:
            if c in X.columns and c in self.cat_categories_:
                cats = self.cat_categories_[c]
                # force to training categories (unknown -> NaN -> all zeros after dummies)
                col = pd.Categorical(X[c].astype("object"), categories=cats)
                dummies = pd.get_dummies(col, prefix=c, prefix_sep='=', dummy_na=False)
                if self.drop_first and dummies.shape[1] > 0:
                    dummies = dummies.iloc[:, 1:]  # drop first category
                X_cat_parts.append(dummies.to_numpy(dtype=float))
        X_cat = np.column_stack(X_cat_parts) if X_cat_parts else np.empty((len(X), 0))

        # 4) concat numeric + categorical
        X_all = np.column_stack([X_num, X_cat]) if X_num.size > 0 or X_cat.size > 0 else np.empty((len(X), 0))

        # 5) add bias as first column
        bias = np.ones((X_all.shape[0], 1), dtype=float)
        X_with_bias = np.hstack([bias, X_all])
        return X_with_bias

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names(self, include_bias=False):
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        if include_bias:
            return np.array(["bias"] + list(self.feature_names_), dtype=object)
        return self.feature_names_.copy()

# Load the trained model from multiple possible paths
MODEL_PATHS = [
    "./car_price.model",  # For Docker deployment
    "./model/car_price.model",  # For root directory local development
    "../model/car_price.model",  # For app directory local development
]

model = None
for MODEL_PATH in MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            break
        except Exception as e:
            print(f"Failed to load model from {MODEL_PATH}. Error: {e}")
            continue

if model is None:
    raise RuntimeError("No valid model found in any of the expected paths")

# Load A2 model package from multiple possible paths
A2_MODEL_PATHS = [
    "./model/best_model.pkl",  # For Docker deployment
    "../model/best_model.pkl",  # For app directory local development
    "./best_model.pkl",  # For root directory local development
    "./notebook/a2_model_complete.pkl",  # Alternative model file
    "../notebook/a2_model_complete.pkl",  # Alternative model file
]

# Try to load A2 model package
a2_model_package = None
a2_model = None
a2_preprocessor = None
a2_features = None

for MODEL_PATH in A2_MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            a2_model_package = joblib.load(MODEL_PATH)
            a2_model = a2_model_package['model']
            a2_preprocessor = a2_model_package['preprocessor']
            a2_features = a2_model_package['features']
            print(f"A2 Model package loaded successfully from {MODEL_PATH}")
            print(f"A2 Package contains: {list(a2_model_package.keys())}")
            break
        except Exception as e:
            print(f"Failed to load A2 model package from {MODEL_PATH}. Error: {e}")
            continue

if a2_model_package is None:
    print("Warning: A2 model package not found or has import issues. A2 page will use original model as fallback.")
    print("This is normal if the model was created in a notebook environment with different imports.")

# Initialize Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for gunicorn if needed later

# Add CSS styling for better UI appearance
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Georgia:wght@400;700&display=swap');
            body {
                font-family: 'Georgia', serif !important;
                margin: 0;
                background-color: #fafafa;
            }
            .nav-bar {
                background-color: #2c3e50;
                padding: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .nav-container {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0 20px;
            }
            .nav-logo {
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-decoration: none;
            }
            .nav-links {
                display: flex;
                gap: 30px;
            }
            .nav-link {
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .nav-link:hover {
                background-color: #34495e;
                text-decoration: none;
                color: white;
            }
            .nav-link.active {
                background-color: #3498db;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            .instruction-card {
                background: white;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create navigation bar component
def create_navbar(current_page):
    return html.Div([
        html.Div([
            html.A("Car Price Predictor", href="/", className="nav-logo"),
            html.Div([
                html.A("Instructions", 
                      href="/", 
                      className=f"nav-link {'active' if current_page == 'home' else ''}"),
                html.A("Predict Price", 
                      href="/predict", 
                      className=f"nav-link {'active' if current_page == 'predict' else ''}"),
                html.A("A2 - Predict Price", 
                      href="/a2-predict", 
                      className=f"nav-link {'active' if current_page == 'a2-predict' else ''}"),
            ], className="nav-links")
        ], className="nav-container")
    ], className="nav-bar")

# Create instructions page layout
def instructions_layout():
    return html.Div([
        create_navbar('home'),
        html.Div([
            html.Div([
                html.H1("Car Price Prediction Assignment System", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
                html.H2("by Dechathon Niamsa-ard [st126235]",
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                
                html.H3("Comparison: Assignment 1 (XGBoost) vs. Assignment 2 (Custom Linear Regression)", 
                       style={'color': '#34495e', 'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("A1 – XGBoost Model", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Strengths:", style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Captures complex, nonlinear relationships between features."),
                            html.Li("Generally achieves higher accuracy (lower MSE, higher R²) on car price prediction."),
                            html.Li("Handles interactions between features automatically.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Limitations:", style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("More of a \"black box\" cause harder to interpret feature effects."),
                            html.Li("Heavier computational cost for training and tuning."),
                            html.Li("Requires more system resources for deployment.")
                        ], style={'color': '#555'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'border': '1px solid #dee2e6'})
                ]),
                
                html.Div([
                    html.H4("A2 – Custom Linear Regression Model", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.P("Strengths:", style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Fully implemented from scratch with gradient descent, regularization options (Ridge, Lasso), and polynomial extensions."),
                            html.Li("Much more interpretable: coefficients directly show how each feature influences car price."),
                            html.Li("Lightweight and efficient — requires less memory and faster to deploy."),
                            html.Li("Includes consistent preprocessing pipeline (imputation, scaling, encoding) and MLflow logging, making it more production-friendly.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Limitations:", style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li("Linear assumption: struggles with nonlinear patterns in car data."),
                            html.Li("Predictive accuracy may be lower than XGBoost when the dataset has complex feature interactions.")
                        ], style={'marginBottom': '15px', 'color': '#555'}),
                        
                        html.P("Best Model Configuration (from experiments):", style={'fontWeight': 'bold', 'color': '#8e44ad', 'marginBottom': '8px'}),
                        html.Div([
                            html.P("LinearRegression-method-sto-lr-0.001-init_method-zeros-momentum-False", 
                                   style={'fontFamily': 'monospace', 'backgroundColor': '#f1f3f4', 'padding': '10px', 'borderRadius': '5px', 'color': '#2c3e50', 'marginBottom': '5px'}),
                            html.P("This configuration uses Stochastic Gradient Descent (SGD) with learning rate 0.001, zero initialization, and no momentum.", 
                                   style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
                        ], style={'backgroundColor': '#f8f4ff', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #d1c4e9'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '30px', 'border': '1px solid #dee2e6'})
                ]),
                
                html.H3("How the Assignment System Works", style={'color': '#34495e', 'marginBottom': '20px'}),
                html.P([
                    "This assignment system demonstrates two different approaches to car price prediction. "
                    "Assignment 1 uses XGBoost (advanced machine learning) while Assignment 2 implements custom linear regression from scratch. "
                    "Both models analyze car specifications, condition, and market trends to provide price estimates for used cars."
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Step-by-Step Instructions", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Ol([
                    html.Li("Navigate to either 'Predict Price' (Assignment 1 - XGBoost) or 'A2 - Predict Price' (Assignment 2 - Custom Linear Regression) using the navigation bar above"),
                    html.Li("Fill in the car details you know in the input form"),
                    html.Li("Don't worry if you don't have all information - you can skip any field"),
                    html.Li("For missing fields, each model uses its own imputation techniques learned during training"),
                    html.Li("Click the 'Predict Price' button to submit your data"),
                    html.Li("Compare the predictions from both models to see how different approaches handle the same data"),
                    html.Li("The predicted price will appear below the form within moments")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Missing Data Handling", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.P([
                    "Both assignment models intelligently handle missing information using their respective preprocessing strategies learned from market data:"
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                html.Ul([
                    html.Li("Assignment 1 (XGBoost): Uses built-in XGBoost preprocessing with median/mean imputation"),
                    html.Li("Assignment 2 (Custom Linear Regression): Uses custom ManualPreprocessor with median for numerical fields and mean for mileage"),
                    html.Li("Numerical fields (Year, Kilometers, Owner, Engine, Power): Uses median from training data"),
                    html.Li("Mileage: Uses mean from training data"),
                    html.Li("Categorical fields (Fuel Type, Transmission, Brand): Uses most frequent from training data"),
                    html.Li("All imputation values are learned from the actual car market data during model training")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Important Notes", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Div([
                    html.P("• Both models are trained specifically on Petrol and Diesel vehicles", 
                           style={'color': '#e74c3c', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("• Predictions are estimates based on historical market data and current trends", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• The more accurate information you provide, the better the prediction", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• Assignment 1 (XGBoost) typically shows higher accuracy but is less interpretable", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• Assignment 2 (Custom Linear Regression) is more interpretable but may have lower accuracy", 
                           style={'color': '#555'})
                ]),
                
                html.Div([
                    html.A("Try Assignment 1 (XGBoost) →", 
                           href="/predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#3498db',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px',
                               'marginRight': '15px'
                           }),
                    html.A("Try Assignment 2 (Custom Linear Regression) →", 
                           href="/a2-predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#e67e22',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px'
                           })
                ], style={'textAlign': 'center'})
                
            ], className="instruction-card")
        ], className="container")
    ])

# Create helper functions for form elements
def labeled_input(label, id_, type_="number", placeholder="", **kwargs):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Input(
            id=id_, 
            type=type_, 
            placeholder=placeholder, 
            style={
                "width": "100%", 
                "padding": "12px", 
                "border": "2px solid #bdc3c7",
                "borderRadius": "5px",
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            },
            **kwargs
        )
    ], style={"marginBottom": "20px"})

def labeled_dropdown(label, id_, options, value=None):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Dropdown(
            id=id_, 
            options=[{"label": o, "value": o} for o in options], 
            value=value, 
            clearable=True,
            placeholder="Select or leave blank for pipeline imputation",
            style={
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            }
        )
    ], style={"marginBottom": "20px"})

# Create prediction page layout
def prediction_layout():
    return html.Div([
        create_navbar('predict'),
        html.Div([
            html.Div([
                html.H1("Car Price Prediction", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. Leave fields blank if you don't have the information.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px', 'fontSize': '16px'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "year", placeholder="e.g., 2016 (leave blank for median from data)", min=1, step=1),
                        labeled_input("Kilometers Driven", "km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Fuel Type", "fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "engine", placeholder="e.g., 1197 (leave blank for median from data)", min=0, step=1),
                        labeled_input("Max Power (bhp)", "power", placeholder="e.g., 82 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Brand", "brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price", 
                           id="predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#4a9d5b",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Create A2 prediction page layout (perfect clone of prediction_layout)
def a2_prediction_layout():
    return html.Div([
        create_navbar('a2-predict'),
        html.Div([
            html.Div([
                html.H1("A2 - Car Price Prediction", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. Leave fields blank if you don't have the information.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px', 'fontSize': '16px'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "a2-year", placeholder="e.g., 2016 (leave blank for median from data)", min=1, step=1),
                        labeled_input("Kilometers Driven", "a2-km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Fuel Type", "a2-fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "a2-transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "a2-owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "a2-mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "a2-engine", placeholder="e.g., 1197 (leave blank for median from data)", min=0, step=1),
                        labeled_input("Max Power (bhp)", "a2-power", placeholder="e.g., 82 (leave blank for median from data)", min=0, step=1),
                        labeled_dropdown("Brand", "a2-brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price", 
                           id="a2-predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#4a9d5b",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="a2-result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Set up main app layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Handle URL routing to display correct page
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/predict':
        return prediction_layout()
    elif pathname == '/a2-predict':
        return a2_prediction_layout()
    else:  # Default to instructions page
        return instructions_layout()

# Handle price prediction when button is clicked
@app.callback(
    Output("result-section", "children"),
    Input("predict", "n_clicks"),
    State("year", "value"),
    State("km", "value"),
    State("fuel", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("power", "value"),
    State("brand", "value"),
)
def predict_price(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price' to get your estimate.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for model prediction
    # Leave missing values as NaN/None - the pipeline will handle imputation
    row = pd.DataFrame([{
        "year": float(year) if year is not None else np.nan,
        "km_driven": float(km) if km is not None else np.nan,
        "fuel": str(fuel) if fuel is not None else None,
        "transmission": str(transmission) if transmission is not None else None,
        "owner": float(owner_num) if owner_num is not None else np.nan,  # Model expects numeric owner
        "engine": float(engine) if engine is not None else np.nan,  # Model expects numeric engine
        "max_power": float(power) if power is not None else np.nan,  # Model expects numeric max_power
        "brand": str(brand) if brand is not None else None,
        "mileage": float(mileage) if mileage is not None else np.nan,  # Model expects numeric mileage
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in row.columns:
        if pd.isna(row.at[0, col]) or row.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction and convert from log scale to price
    try:
        # Debug information for troubleshooting
        print("Input row shape:", row.shape)
        print("Input row columns:", list(row.columns))
        print("Input row values:", row.iloc[0].to_dict())
        print("Imputed fields:", imputed_fields)
        
        pred_log = float(model.predict(row)[0])
        price = float(np.exp(pred_log))
        
        print(f"Predicted log price: {pred_log:.4f}")
        print(f"Predicted price: {price:.2f}")
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Estimated Price: {price:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '32px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by Pipeline", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the trained pipeline's imputation strategy:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → pipeline default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': '3px solid #27ae60'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Model type: {type(model)}")
        return html.Div([
            html.H3("Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Handle A2 price prediction when button is clicked
@app.callback(
    Output("a2-result-section", "children"),
    Input("a2-predict", "n_clicks"),
    State("a2-year", "value"),
    State("a2-km", "value"),
    State("a2-fuel", "value"),
    State("a2-transmission", "value"),
    State("a2-owner", "value"),
    State("a2-mileage", "value"),
    State("a2-engine", "value"),
    State("a2-power", "value"),
    State("a2-brand", "value"),
)
def a2_predict_price(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price' to get your estimate.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for A2 model prediction using the model package approach
    sample_data = pd.DataFrame([{
        'year': float(year) if year is not None else np.nan,
        'km_driven': float(km) if km is not None else np.nan,
        'fuel': str(fuel) if fuel is not None else None,
        'transmission': str(transmission) if transmission is not None else None,
        'brand': str(brand) if brand is not None else None,
        'owner': float(owner_num) if owner_num is not None else np.nan,
        'engine': float(engine) if engine is not None else np.nan,
        'power': float(power) if power is not None else np.nan,
        'mileage': float(mileage) if mileage is not None else np.nan,
        'max_power': float(power) if power is not None else np.nan,  # Note: power and max_power are the same
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in sample_data.columns:
        if pd.isna(sample_data.at[0, col]) or sample_data.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction using the A2 model package approach
    try:
        # Debug information for troubleshooting
        print("A2 Input data shape:", sample_data.shape)
        print("A2 Input data columns:", list(sample_data.columns))
        print("A2 Input data values:", sample_data.iloc[0].to_dict())
        print("A2 Imputed fields:", imputed_fields)
        
        # Use A2 model if available, otherwise fall back to original model
        if a2_model is not None and a2_preprocessor is not None and a2_features is not None:
            print("A2 Features expected by preprocessor:", a2_features)
            
            # Preprocess the sample data using the A2 preprocessor
            sample_data_transformed = a2_preprocessor.transform(sample_data[a2_features])
            print("A2 Transformed data shape:", sample_data_transformed.shape)
            
            # Make prediction using the A2 model
            try:
                # Try with is_polynomial=False first (as in notebook)
                predicted_log_price = a2_model.predict(sample_data_transformed, is_polynomial=False)
            except TypeError:
                try:
                    # Try with is_polynomial=True
                    predicted_log_price = a2_model.predict(sample_data_transformed, is_polynomial=True)
                except TypeError:
                    # Try without parameter
                    predicted_log_price = a2_model.predict(sample_data_transformed)
            
            predicted_price = np.expm1(predicted_log_price)  # Inverse of log1p transformation
            price = float(predicted_price[0])
            
            print(f"A2 Predicted log price: {predicted_log_price[0]:.4f}")
            print(f"A2 Predicted price: {price:.2f}")
        else:
            # Fallback to original model approach
            print("A2 Model package not available, using original model")
            pred_log = float(model.predict(sample_data)[0])
            price = float(np.exp(pred_log))
            
            print(f"A2 Fallback predicted log price: {pred_log:.4f}")
            print(f"A2 Fallback predicted price: {price:.2f}")
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Estimated Price: {price:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '32px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "power": "Max Power → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by A2 Preprocessor", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the A2 model's preprocessor imputation strategy:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → A2 preprocessor default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': '3px solid #27ae60'
        })
        
    except Exception as e:
        print(f"A2 Prediction error: {e}")
        print(f"A2 Model type: {type(a2_model) if a2_model is not None else 'None'}")
        print(f"A2 Preprocessor type: {type(a2_preprocessor) if a2_preprocessor is not None else 'None'}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H3("A2 Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Start the application
if __name__ == "__main__":
    # Get configuration from environment variables
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    print(f"Starting Car Price Prediction App on port {port}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
