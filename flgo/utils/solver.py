import numpy as np
import torch


class NormSolver:
    MAX_ITER = 1000  # 最大迭代次数
    STOP_CRIT = 1e-5  # 停止准则
    Min_Max = True  # True for min, False for max

    @staticmethod
    def _norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min/max_{c} |c*x1 + (1-c)*x2|_2^2
        d is the distance (objective) optimized
        v1v1 = <x1, x1>
        v1v2 = <x1, x2>
        v2v2 = <x2, x2>
        """
        if NormSolver.Min_Max:
            # Min norm case (same as original)
            if v1v2 >= v1v1:
                # Case: Fig 1, third column (min)
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                # Case: Fig 1, first column (min)
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            # Case: Fig 1, second column (min)
            gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
            cost = v2v2 + gamma * (v1v2 - v2v2)
            return gamma, cost
        else:
            # Max norm case (reverse logic)
            if v1v2 <= v1v1:
                # Case: Fig 1, third column (max)
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 <= v2v2:
                # Case: Fig 1, first column (max)
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            # Case: Fig 1, second column (max)
            gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
            cost = v2v2 + gamma * (v1v2 - v2v2)
            return gamma, cost

    @staticmethod
    def _norm_2d(vecs, dps):
        """
        Find the norm solution as combination of two points
        This is correct only in 2D
        ie. min/max_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_i >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[j])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().item()
                c, d = NormSolver._norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _norm_2d_accelerated(vecs, dps):
        """
        Accelerated version of _norm_2d using matrix multiplication
        """
        dmin = 1e8
        # Stack vectors and compute M = V * V^T
        vec = torch.stack(vecs)
        M = torch.matmul(vec, vec.T).cpu().numpy()

        sol = None
        for i in range(len(vecs)):
            if (i, i) not in dps:
                dps[(i, i)] = M[i, i]
            for j in range(i + 1, len(vecs)):
                if (j, j) not in dps:
                    dps[(j, j)] = M[j, j]
                if (i, j) not in dps:
                    dps[(i, j)] = dps[(j, i)] = M[i, j]
                c, d = NormSolver._norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y - z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / grad[proj_grad > 0]

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = NormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the norm element in the convex hull
        as min/max |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = NormSolver._norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0
        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < NormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = NormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = NormSolver._norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < NormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

    @staticmethod
    def find_norm_element_FW(vecs, minmax=True):
        """
        Given a list of vectors (vecs), this method finds the norm element in the convex hull
        as min/max |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        """
        # Solution lying at the combination of two points
        NormSolver.Min_Max = minmax
        dps = {}
        init_sol, dps = NormSolver._norm_2d_accelerated(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        for _ in range(NormSolver.MAX_ITER):
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = NormSolver._norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < NormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

        return sol_vec, nd