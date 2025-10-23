#!/usr/bin/env conda run -n robotics_course python
import numpy as np
import osqp
import scipy.sparse as sparse


class Controller:
    def __init__(self, K, gamma, E, w_max, u_max, alpha, K_e, n_x = 1, n_p = 7, dt=0.01):
        self.dt = dt

        self.K = K
        self.K_e = K_e
        self.gamma = gamma
        self.E = E
        self.w_max = w_max.reshape(n_p, 1)
        self.u_max = u_max.reshape(n_x, 1)
        self.n_x = n_x                                      # Number of states (equal to number of inputs)
        self.n_p = n_p                                      # Number of parameters
        self.alpha = alpha
        # self.Kp = np.diag([350, 250, 100, 50, 50, 50])      # P gain
        # self.Kd = 0.1*np.diag([350, 250, 100, 50, 50, 50])     # D gain
        self.Kp = 1*np.diag([5, 15, 10, 2])
        self.Kd = 0.005 * self.Kp

        self.x = np.zeros((n_x, 1))
        self.x_d = np.zeros((n_x, 1))
        # self.w = np.array([[np.random.uniform(-self.w_max[i, 0], self.w_max[i, 0])] for i in range(n_p)])
        self.w = np.zeros((n_p, 1))                         # Initial parameter estimate
        # self.w = np.ones((n_p, 1)) * 1                     # Initial parameter estimate

        self.qp = osqp.OSQP()

        # Preallocate variables for W bounds
        self.int_err_sq = 0.0
        gamma_inv = np.linalg.inv(self.gamma)
        self.lambda_min_inv_gamma = np.linalg.eigvalsh(gamma_inv).min()
        self.lambda_max_inv_gamma = np.linalg.eigvalsh(gamma_inv).max()

        # Calculate worst case V(0)
        e0 = np.array([[0.0]]) 
        Va_0 = 0.5 * e0.T @ e0
        W_init_max = (np.linalg.norm(self.w_max, 2) + np.linalg.norm(self.w, 2))**2
        unknown_V0_part = 0.5 * self.lambda_max_inv_gamma * W_init_max**2
        self.V_max_0 = Va_0 + unknown_V0_part

        # Keep track of trajectories for plotting
        self.e_traj = []
        self.w_traj = []
        self.joint_traj = []
        self.vel_traj = []
        self.cbf_traj = []
        self.u_traj = []
        self.slack_traj = []

        pass

    def u_pd(self, x_d, xd_d, time):
        """
        PD controller for testing the simulation.
        Paraameters: desired x, desired xdot
        Returns: joint torques
        """
        x_d = x_d.reshape(self.n_x, 1)
        xd_d = xd_d.reshape(self.n_x, 1)
        u = (self.Kp @ (x_d - self.x) + self.Kd @ (xd_d - self.x_d))
        self.u_traj.append(u.flatten())
        self.e_traj.append((x_d - self.x).flatten())
        return u.flatten().tolist()

    def u_(self, x_d, xd_d, time):
        """
        Simple Sliding mode Adaptive controller for testing the simulation.
        Paraameters: desired x, desired xdot, current simulation time
        Returns: joint torques
        """
        x_d = x_d.reshape(self.n_x, 1)
        xd_d = xd_d.reshape(self.n_x, 1)

        K = 0.01 * np.diag([5, 15, 10, 2]) # Kd gain
        gamma = 5 * np.diag([5, 15, 10, 2]) @ np.linalg.inv(K) #gamma * K is Kp

        r = (self.x_d - xd_d) + gamma @ (self.x - x_d)
        # self.w = self.w + self.dt * (-np.linalg.inv(self.gamma) @ self.z(time).T @ r)
        # u = self.z(time) @ self.w - self.K @ r
        rho = 5
        mu = 0.5
        if np.linalg.norm(self.z(time).T @ r, 2) > mu:
            deltaw = -rho * (self.z(time).T @ r)/np.linalg.norm(self.z(time).T @ r, 2)
        else:
            deltaw = -rho * (self.z(time).T @ r)/mu
        w_hat =  self.w + deltaw
        u = self.z(time) @ w_hat - K @ r
        
        return u.flatten().tolist()
    
    def u(self, x_d, xd_d, time):
        """
        Adaptive control law based on the most recent parameter estimate.
        
        Parameters
        ----------
        x_d : np.ndarray
            Desired state.
        xd_d : np.ndarray
            Desired state derivative.
        time : float
            Current time.
            
        Returns
        -------
        np.ndarray(self.nx, 1)
            Control input.
        """
        x_d = x_d.reshape(self.n_x, 1)
        xd_d = xd_d.reshape(self.n_x, 1)
        error = (self.x - x_d)
        self.int_err_sq += (error.T @ error).item() * self.dt
        self.update(error, time)                              # Updates the parameter vector based on the error
        delta_w = self.cbf_qp(error, xd_d, time)              # safety modification for parameter
        if delta_w is None:
            print("\nERROR\nQP failed, using zero delta_w\n")
            delta_w = np.zeros((self.n_p, 1))
            u = self.saturate(self.z(time) @ self.w + xd_d - self.K @ error)
        else:
            u = self.z(time) @ (self.w + delta_w) + xd_d - self.K @ error
        self.w_traj.append(self.w.flatten())
        self.u_traj.append(u.flatten())
        self.joint_traj.append(self.x.flatten())
        self.vel_traj.append(self.x_d.flatten())
        self.e_traj.append(error.flatten())
        return u.flatten().tolist()
    
    def update(self, error, time):
        """
        Update the parameter estimate.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """
        self.w = self.w - self.dt * (self.gamma @ self.z(time).T @ self.v_a_partial_e(error).T)
        pass

    def w_max_bound(self):
        """
        Calculates the rigorous, time-decaying bound W_bound,d(t).
        This method should be called before the QP in every time step.
        """
        # Calculate the decreasing upper bound on V(t)
        v_t_upper_bound = self.V_max_0 - self.K_e * self.int_err_sq
        
        # Ensure the argument of the square root is non-negative
        numerator = 2 * max(0, v_t_upper_bound)
        
        # Calculate the final dynamic bound
        w_bound_d = np.sqrt(numerator / self.lambda_min_inv_gamma)
        
        return w_bound_d
    
    def cbf_qp(self, error, xd_d, time):
        """
        Solve the QP problem to find the safety modification for the parameter.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        np.ndarray(self._np, 1)
            Safety modification for the parameter.
        """
        # print(f"CBF value: {self.cbf()}")
        # print(f"CBF gradient: {self.cbf_grad()}")
        P = sparse.diags(np.append(np.repeat(1, self.n_p), np.array([10, 600])), format='csc')
        # P = sparse.diags(np.ones(self.n_p) * 10, format='csc')
        q = np.zeros((self.n_p + 2, 1))
        # q = np.zeros(self.n_p)
        A_clf = np.hstack((self.v_a_partial_e(error) @ self.z(time), np.array([[-1, 0]])))
        A_effort = np.hstack((self.z(time), np.zeros((self.n_x, 2))))
        A_cbf = np.hstack((self.cbf_grad() @ self.z(time), np.array([[0, -1]])))
        A = sparse.vstack([sparse.csr_matrix(A_clf), sparse.csr_matrix(A_effort), sparse.csr_matrix(A_cbf)], format='csc')
        # A = sparse.vstack([sparse.csr_matrix(A_clf), sparse.csr_matrix(A_effort)], format='csc')
        l_clf = np.array(-np.inf)
        l_effort = -self.u_max - self.z(time) @ self.w - xd_d + self.K @ error
        # l_cbf = -self.alpha * self.cbf() + self.cbf_grad() @ (self.K @ error - xd_d) + np.linalg.norm(self.cbf_grad() @ self.z(time), 2) * (np.linalg.norm(self.w, 2) + np.linalg.norm(self.w_max, 2)) + np.linalg.norm(self.cbf_grad(), 2) * self.E
        l_cbf = -self.alpha * self.cbf() + np.linalg.norm(self.cbf_grad() @ self.z(time), 2) * self.w_max_bound() + self.cbf_grad() @ (self.K @ error - xd_d) + np.linalg.norm(self.cbf_grad(), 2) * self.E
        l = np.vstack([l_clf, l_effort, l_cbf]) 
        # l = np.vstack([l_clf, l_effort, ]) 
        u_clf = -self.alpha_3(error) + self.v_a_partial_e(error) @ (self.K @ error + self.z(time) @ self.gamma.T @ self.v_a_partial_w(error).T) - np.linalg.norm(self.v_a_partial_e(error), 2) * self.E
        u_effort = self.u_max - self.z(time) @ self.w - xd_d + self.K @ error
        u_cbf = np.array(np.inf)
        u = np.vstack([u_clf, u_effort, u_cbf])
        # u = np.vstack([u_clf, u_effort, ])
        # print(f"A shape: {A.shape}, l shape: {l.shape}, u shape: {u.shape}")
        qp = osqp.OSQP()
        qp.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        result = qp.solve()
        self.cbf_traj.append(self.cbf())
        self.slack_traj.append(result.x[-2:])
        if result.info.status == 'solved':
            return result.x[:self.n_p].reshape(self.n_p, 1)
        else:
            print(f"[{time}] QP solver failed with status: {result.info.status}")
            return None

    def saturate(self, u):
        """
        Saturate the control input to the maximum allowed values.
        
        Parameters
        ----------
        u : np.ndarray(self.nx, 1)
            Control input.
            
        Returns
        -------
        np.ndarray(self.nx, 1)
            Saturated control input.
        """
        return np.clip(u, -self.u_max, self.u_max)
    
    def z(self, time):
        """
        Basis function for regressor based formulation.
        psi = Z(x, t) * W
        
        Parameters
        ----------
        time : float
            Current time.
            
        Returns
        -------
        np.ndarray(self.nx, self._np)
            Control input.
        """
        raise NotImplementedError("z method must be implemented by subclass")
    
    def cbf(self):
        """
        Template function for Control Barrier Function (CBF).
        Should be overridden by subclasses.
        
        Parameters
        ----------
        None
        (state and estimated parameter are availbale in self.x and self.w respectively.)
            
        Returns
        -------
        float 
            CBF value(s) indicating constraint satisfaction.
        """
        raise NotImplementedError("CBF method must be implemented by subclass")
    
    def cbf_grad(self):
        """
        Template function for the gradient of the CBF.
        Should be overridden by subclasses.
        
        Parameters
        ----------
        None
        (state and estimated parameter are availbale in self.x and self.w respectively.)
            
        Returns
        -------
        np.ndarray(self.nx, self._np)
            Gradient of the CBF.
        """
        raise NotImplementedError("CBF_grad method must be implemented by subclass")
    
    def v_a_partial_e(self, error):
        """
        Template function for the partial derivative of adaptive Lyapunov function.
        Should be overridden by subclasses.
        
        Parameters
        ----------
        error : np.ndarray(self.nx, 1)
            Error between current state and desired state.
            
        Returns
        -------
        np.ndarray(self.nx, 1)
            Adaptive lyapunov function.
        """
        raise NotImplementedError("v_a method must be implemented by subclass")
    
    def v_a_partial_w(self, error):
        """
        Template function for the partial derivative of adaptive Lyapunov function.
        Should be overridden by subclasses.
        
        Parameters
        ----------
        error : np.ndarray(self.nx, 1)
            Error between current state and desired state.
            
        Returns
        -------
        np.ndarray(self.nw, 1)
            Adaptive lyapunov function.
        """
        raise NotImplementedError("v_a method must be implemented by subclass")

    def alpha_3(self, error):
        """
        Template function for the alpha_3 function.
        Should be overridden by subclasses.
        
        Parameters
        ----------
        error : np.ndarray(self.nx, 1)
            Error between current state and desired state.
            
        Returns
        -------
        float
            Alpha_3 value.
        """
        raise NotImplementedError("alpha_3 method must be implemented by subclass")