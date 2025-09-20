from scipy import integrate
import numpy as np


# FitzHugh–Nagumo parameters (original scale)
TAU = 0.02
I_EXT = 0.5
A = 0.7
B = 0.8

# Rescaling coefficients (see highlighted paper snippet)
ALPHA = 10.0  # state scaling so |v'| = O(1)
BETA = 200.0  # time scaling to shrink horizon to [0, 1]


def FHN_rhs_rescaled(tau, y_tilde):
    """Right-hand side of the rescaled FitzHugh–Nagumo system.

    The solver works with the scaled state ``y_tilde = alpha * y`` and
    the compressed time variable ``tau = t / beta``. We therefore map
    back to the original state before evaluating the classical vector
    field and then reapply the scaling (alpha * beta) to obtain the
    derivative with respect to ``tau``.
    """

    # Recover original state variables y = (v, w)
    v = y_tilde[0] / ALPHA
    w = y_tilde[1] / ALPHA

    dv_dt = v - (v ** 3) / 3.0 - w + I_EXT
    dw_dt = TAU * (v + A - B * w)

    # Convert derivative to the scaled coordinates / time variable
    dv_dt_tilde = ALPHA * BETA * dv_dt
    dw_dt_tilde = ALPHA * BETA * dw_dt

    return np.array([dv_dt_tilde, dw_dt_tilde])


def get_data(N, T=500):
    data_x = []
    data_y = []

    tau_grid = np.linspace(0.0, 1.0, T + 1)

    for _ in range(N):
        c0 = np.random.uniform(-1.0, 1.0)
        # Initial condition in the scaled coordinates y_tilde(0) = alpha * y(0)
        x0_tilde = np.array([ALPHA * c0, 0.0])

        sol = integrate.solve_ivp(
            FHN_rhs_rescaled,
            [0.0, 1.0],
            x0_tilde,
            t_eval=tau_grid,
            vectorized=False,
        )

        # Use the scaled voltage component (first entry)
        v_series = sol.y[0]
        data_x.append(v_series[:-1])
        data_y.append(v_series[1:])

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x.reshape(N, T, 1), data_y.reshape(N, T, 1)
