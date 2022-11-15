import numpy as np
from abc import ABC, abstractmethod

class Tensor(object):
    def __init__(self, data, requires_grad=False):
        self.__val = np.array(data)
        self.__grad = np.zeros(self.__val.shape)
        self.__lastop = None
        self.__requires_grad = requires_grad
    
    def reset(self):
        self.__lastop = None

    def zero_grad(self):
        self.__grad = np.zeros(self.__val.shape)

    @property
    def val(self):
        return self.__val
    
    @val.setter
    def val(self, new_val):
        self.__val = new_val
    
    @property
    def lastop(self):
        return self.__lastop
    
    @lastop.setter
    def lastop(self, new_op):
        self.__lastop = new_op
    
    @property
    def requires_grad(self):
        return self.__requires_grad
    
    @property
    def grad(self):
        return self.__grad
    
    @grad.setter
    def grad(self, x):
        self.__grad = np.array(x)

    def __sub__(self, x):
        sub = Sub(self, x)
        ans = Tensor(self.val - x.val, requires_grad=self.requires_grad or x.requires_grad)
        sub.ans = ans
        ans.lastop = sub
        return ans
    
    def __matmul__(self, x):
        matmul = MatMul(self, x)
        ans = Tensor(self.val @ x.val, requires_grad=self.requires_grad or x.requires_grad)
        matmul.ans = ans
        ans.lastop = matmul
        return ans
    
    def __pow__(self, p):
        pow = Pow(self, p)
        ans = Tensor(self.val ** p, requires_grad=self.requires_grad)
        pow.ans = ans
        ans.lastop = pow
        return ans
    
    def sigmoid(self):
        sigmoid = Sigmoid(self)
        ans = Tensor(1.0/(1.0 + np.exp(-self.val)), requires_grad=self.requires_grad)
        sigmoid.ans = ans
        ans.lastop = sigmoid
        return ans
    
    def tanh(self):
        tanh = Tanh(self)
        ans = Tensor(np.tanh(self.val), requires_grad=self.requires_grad)
        tanh.ans = ans
        ans.lastop = tanh
        return ans
    
    # back propagate the grad----coef
    def backward(self, coef=None):
        if self.lastop:

            if coef is None:
                coef = np.ones(self.val.shape)

            self.lastop.coef = coef
            self.lastop.backward()

class Operator(ABC):
    def __init__(self):
        self._opnum = []
        self.__ans = None
        self.__coef = None
    
    def reset(self):
        self._opnum.clear()
        self.__ans = None

    @abstractmethod
    def backward(self):
        pass
    
    @property
    def ans(self):
        return self.__ans

    @ans.setter
    def ans(self, ans):
        self.__ans = ans
    
    @property
    def coef(self):
        return self.__coef

    @coef.setter
    def coef(self, coef):
        self.__coef = coef
    
class Sub(Operator):
    def __init__(self, opnum1, opnum2):
        super().__init__()
        self._opnum.append(opnum1)
        self._opnum.append(opnum2)
    
    def backward(self):
        if self._opnum[0].requires_grad:
            grad = np.ones(self._opnum[0].val.shape) * self.coef
            self._opnum[0].grad += grad
            self._opnum[0].backward(grad)
        if self._opnum[1].requires_grad:
            grad = -np.ones(self._opnum[1].val.shape) * self.coef
            self._opnum[1].grad += grad
            self._opnum[1].backward(grad)
        
        self.reset()

class MatMul(Operator):
    def __init__(self, opnum1, opnum2):
        super().__init__()
        self._opnum.append(opnum1)
        self._opnum.append(opnum2)
    
    def backward(self):
        if self._opnum[0].requires_grad:
            A = np.kron(self.coef, np.ones(self._opnum[0].val.shape))
            B = np.expand_dims(self._opnum[1].val.flatten('F'), -2)
            mask = np.eye(self._opnum[0].val.shape[-2]).flatten()
            mask = np.expand_dims(mask, -1)
            C = np.kron(mask, B)
            D = A * C
            point = np.arange(self._opnum[0].val.shape[-1], self.ans.val.shape[-1] * self._opnum[0].val.shape[-1], self._opnum[0].val.shape[-1])
            grad = np.split(D, point, axis=1)
            grad = sum(grad)
            point = np.arange(self._opnum[0].val.shape[-2], self._opnum[0].val.shape[-2] * self.ans.val.shape[-2], self._opnum[0].val.shape[-2])
            grad = np.split(grad, point, axis=0)
            grad = sum(grad)
            self._opnum[0].grad += grad
            self._opnum[0].backward(grad)
        if self._opnum[1].requires_grad:
            A = np.kron(self.coef, np.ones(self._opnum[1].val.shape))
            B = np.expand_dims(self._opnum[0].val.flatten(), -1)
            mask = np.eye(self._opnum[1].val.shape[-1]).flatten()
            mask = np.expand_dims(mask, -2)
            C = np.kron(B, mask)
            D = A * C
            point = np.arange(self._opnum[1].val.shape[-1], self._opnum[1].val.shape[-1] * self.ans.val.shape[-1], self._opnum[1].val.shape[-1])
            grad = np.split(D, point, axis=1)
            grad = sum(grad)
            point = np.arange(self._opnum[1].val.shape[-2], self._opnum[1].val.shape[-2] * self.ans.val.shape[-2], self._opnum[1].val.shape[-2])
            grad = np.split(grad, point, axis=0)
            grad = sum(grad)
            self._opnum[1].grad += grad
            self._opnum[1].backward(grad)

        self.reset()

class Pow(Operator):
    def __init__(self, opnum, p):
        super().__init__()
        self._opnum.append(opnum)
        self.__p = np.array(p)

    def backward(self):
        t = self.__p * np.power(self._opnum[0].val, self.__p - 1)
        grad = self.coef * t
        self._opnum[0].grad += grad
        self._opnum[0].backward(grad)
        self.reset()

class Sigmoid(Operator):
    def __init__(self, opnum):
        super().__init__()
        self._opnum.append(opnum)
    
    def backward(self):
        grad = self.ans.val * (1.0 - self.ans.val) * self.coef
        self._opnum[0].grad += grad
        self._opnum[0].backward(grad)
        self.reset()


class Tanh(Operator):
    def __init__(self, opnum1):
        super().__init__()
        self._opnum.append(opnum1)

    def backward(self):
        grad = (1.0 - self.ans.val * self.ans.val) * self.coef
        self._opnum[0].grad += grad
        self._opnum[0].backward(grad)
        self.reset()