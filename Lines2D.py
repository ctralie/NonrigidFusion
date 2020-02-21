import numpy as np
import matplotlib.pyplot as plt

def clamp(arrs, boolexpr):
    """
    Clamp numpy arrays to a boolean expression
    Parameters
    ----------
    arrs: Length-k list of [ndarray(N)]
        List of arrays
    boolexpr: ndarray(N, dtype=np.boolean)
        Boolean expression
    Returns
    -------
    Length-k list of [ndarray(M)], where M is
    the number of true evaluations in boolexpr
    """
    return [arr[boolexpr] for arr in arrs]

def ray_intersect_loop(p0, v, X):
    """
    Parameters
    ----------
    p0: ndarray(2)
        Initial point on ray
    v: ndarray(2)
        Direction of ray
    X: ndarray(N, 2)
        Points on the loop
    
    Returns {'hit':boolean,
                    Whether an intersection happened,
            'p': ndarray(2)
                    Point of intersection,
             'n': ndarray(2)
                    Normal at the point of intersection
            }
    """
    # vec{a} + s*vec{u} refers to line segments on curve
    # vec{b} + t*vec{v} refers to ray
    A = X
    B = np.roll(X, -1, axis=0)
    u = B - A
    # Use Cramer's rule
    # uy*vx - ux*vy
    denom = u[:, 1]*v[0] - u[:, 0]*v[1] 
    # vx*(by - ay) - vy*(bx - ax)
    num_s = v[0]*(p0[1]-A[:, 1]) - v[1]*(p0[0]-A[:, 0])
    # ux*(by - ay) - uy*(bx - ax)
    num_t = u[:, 0]*(p0[1]-A[:, 1]) - u[:, 1]*(p0[0]-A[:, 0])
    
    idx = np.arange(X.shape[0])
    [num_s, num_t, denom, idx] = clamp([num_s, num_t, denom, idx], np.abs(denom) > 0)
    s = num_s/denom
    t = num_t/denom
    # Intersection is within the bounds of the segments
    hit = False
    p = np.zeros(2)
    n = np.zeros(2)
    [s, t, idx] = clamp([s, t, idx], (s >= 0)*(s <= 1))
    if s.size > 0:
        [s, t, idx] = clamp([s, t, idx], t >= 0)
        if t.size > 0:
            minidx = np.argmin(t)
            s = s[minidx]
            t = t[minidx]
            idx = idx[minidx]
            p = p0 + t*v
            n = u[idx, :]
            n = np.array([-n[1], n[0]])
            n = n/np.sqrt(np.sum(n**2))
            hit = True
    return {'hit':hit, 'p':p, 'n':n}

if __name__ == '__main__':
    X = np.array([[-1, 0.5], [1, 0], [-1, 1]])
    p0 = np.array([-1, -1])
    v = np.array([1, 1])
    Y = np.zeros((X.shape[0]+1, 2))
    Y[0:X.shape[0], :] = X
    Y[-1, :] = X[0, :]
    plt.plot(Y[:, 0], Y[:, 1])
    plt.scatter(p0[0], p0[1])
    plt.plot([p0[0], p0[0]+v[0]], [p0[1], p0[1]+v[1]])

    res = ray_intersect_loop(p0, v, X)
    if res['hit']:
        p = res['p']
        print(p)
        plt.scatter(p[0], p[1])
        plt.plot([p0[0], p[0]], [p0[1], p[1]])
    plt.show()