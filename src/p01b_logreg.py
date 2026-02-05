import numpy as np
import util

from linear_model import LinearModel


    

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])
        def sigmoid(x, t1, t2):
            z = (t1*x+t2).astype("float_")
            return 1.0/(1.0+np.exp(-z))

        def log_likelihood(x, y, t1, t2):
            sigmoid_probs = sigmoid(x, t1, t2)
            return np.sum(y*np.log(sigmoid_probs)+(1-y)*np.log(1-sigmoid_probs))

        def gradient(m, x, y, t1, t2):
            sigmoid_probs = sigmoid(x, t1, t2)
            return np.array([[np.sum((y-sigmoid_probs)*x), np.sum((y-sigmoid_probs))]])
        def hessian(x, y, t1, t2): #v^2
            m = np.reshape(x.shape[0],1)
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        # *** START CODE HERE ***
        # *** END CODE HERE ***
