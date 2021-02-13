# coding: utf-8
import math
import numpy as np

EPS = 1e-6


class Float3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.v = np.array([x, y, z], dtype=float)
    
    def __neg__(self):
        return Float3(*(-self.v))
    
    def __add__(self, rhs):
        return Float3(*(self.v + rhs.v))
    
    def __sub__(self, rhs):
        return Float3(*(self.v - rhs.v))
    
    def __mul__(self, rhs):
        return Float3(*np.multiply(self.v, rhs.v))
    
    def add(self, rhs: float):
        return Float3(*(self.v + rhs))
    
    def sub(self, rhs: float):
        return Float3(*(self.v - rhs))
    
    def mul(self, rhs: float):
        return Float3(*(self.v * rhs))
    
    def div(self, rhs: float):
        return Float3(*(self.v / rhs))
    
    def x(self): return self.v[0]
    def y(self): return self.v[1]
    def z(self): return self.v[2]
    def xyz(self): return self.v[:]

    def r(self): return self.v[0]
    def g(self): return self.v[1]
    def b(self): return self.v[2]
    def rgb(self): return self.v[:]

    def clone(self):
        return Float3(*self.v)

    def dot(self, other):
        return np.dot(self.v, other.v)
    
    def cross(self, other):
        return Float3(*np.cross(self.v, other.v))
    
    def normalize(self):
        norm = self.length_squared()
        if norm == 0:
            return self.v
        ret = self.v / np.sqrt(norm)
        return Float3(*ret)

    def length(self):
        return np.linalg.norm(self.v)
    
    def length_squared(self):
        return np.sum(self.v**2)
    
    def sqrt(self):
        return Float3(*np.sqrt(self.v))
    
    def saturate(self):
        return Float3(*np.clip(self.v, 0., 1.))
    
    def near_zero(self):
        return self.v < EPS
    
    def reflect(self, normal):
        ret = self.v - 2. * self.dot(normal) * normal.v
        return Float3(*ret)
    
    # def refract(self, normal, etai_over_etat):
    #     cos_theta = np.min(1., (-self).dot(normal))
    #     r_perp = (self + normal.mul(cos_theta)).mul(etai_over_etat)
    #     r_paral = -normal.mul((1. - np.sqrt(np.abs(r_perp.length_squared()))))
    #     return r_perp + r_paral

    def refract(self, normal, etai_over_etat):
        uv = self.normalize()
        dt = uv.dot(normal)
        d = 1. - etai_over_etat**2. * (1. - dt**2.)
        if d > 0.:
            return (uv - normal.mul(dt)).mul(-etai_over_etat) - normal.mul(math.sqrt(d))
        else:
            return None
    
    def as_u8(self):
        x, y, z = self.saturate().xyz()
        return [int(x * 255.), int(y * 255.), int(z * 255.)]
    
    def as_bgr8(self):
        z, y, x = self.saturate().xyz()
        return [int(x * 255.), int(y * 255.), int(z * 255.)]


        x, y, z = self.saturate().xyz()
        return [int(x * 255.), int(y * 255.), int(z * 255.)]
    
    def lerp(self, other, t):
        return (other - self).mul(t) + self
    
    def gamma(self, factor=2.2):
        recip_factor = 1. / factor
        return Float3(*(self.v**recip_factor))
    
    def degamma(self, factor=2.2):
        return Float3(*(self.v**factor))
    
    @classmethod
    def from_rgb(cls, r, g, b):
        return cls(float(r) / 255., float(g) / 255., float(b) / 255.)

    @classmethod
    def one(cls):
        return cls(*np.ones(3))
    
    @classmethod
    def full(cls, value):
        return cls(*np.full(3, value))
    
    @classmethod
    def xaxis(cls):
        return cls(1.0, 0.0, 0.0)
    
    @classmethod
    def yaxis(cls):
        return cls(0.0, 1.0, 0.0)
    
    @classmethod
    def zaxis(cls):
        return cls(0.0, 0.0, 1.0)
    
    @classmethod
    def random(cls):
        return cls(*np.random.rand(3))
    
    @classmethod
    def random_limit(cls, min_value, max_value):
        range_value = max_value - min_value
        r = np.random.rand(3)
        return cls(
            r[0] * range_value + min_value,
            r[1] * range_value + min_value,
            r[2] * range_value + min_value)
    
    @classmethod
    def random_in_unit_sphere(cls):
        while True:
            point = Float3.random_limit(-1, 1)
            if point.length_squared() < 1:
                return point
    
    @classmethod
    def random_unit_vector(cls):
        return cls.random_in_unit_sphere().normalize()
    
    @classmethod
    def random_in_hemisphere(cls, normal):
        in_unit_sphere = cls.random_in_unit_sphere()
        if in_unit_sphere.dot(normal) > 0:
            return in_unit_sphere
        else:
            return -in_unit_sphere

    @classmethod
    def random_in_unit_disk(cls):
        while True:
            x, y, _ = cls.random_limit(-1., 1.).xyz()
            p = cls(x, y, 0)
            if p.length_squared() < 1.:
                return p
    
    @classmethod
    def random_cosine_direction(cls):
        r1, r2 = np.random.rand(2)
        z = np.sqrt(1. - r2)
        phi = r1 * 2. * math.pi
        x = math.cos(phi) * np.sqrt(r2)
        y = math.sin(phi) * np.sqrt(r2)
        return cls(x, y, z)
    
    @classmethod
    def random_to_sphere(cls, radius, distance_squared):
        rnd = cls.random()
        rr = min(radius**2., distance_squared)
        cos_theta_max = math.sqrt(1. - rr * (1. / distance_squared))
        z = 1. - rnd.y() * (1. - cos_theta_max)
        sqrtz = math.sqrt(1. - z**2.)
        phi = rnd.x() * 2. * math.pi
        x = math.cos(phi) * sqrtz
        y = math.sin(phi) * sqrtz
        return cls(x, y, z)


Vec3 = Float3
Color = Float3
Point3 = Float3


class Quat:
    def __init__(self, x: 0., y: 0., z: 0., w: 1.):
        self.q = np.array([x, y, z, w], dtype=float)
    
    def __neg__(self):
        return Quat(*(-self.q))

    def __mul__(self, rhs):
        return Quat(
            self.w() * rhs.x() + self.x() * rhs.w() + self.y() * rhs.z() - self.z() * rhs.y(),
            self.w() * rhs.y() + self.y() * rhs.w() + self.z() * rhs.x() - self.x() * rhs.z(),
            self.w() * rhs.z() + self.z() * rhs.w() + self.x() * rhs.y() - self.y() * rhs.x(),
            self.w() * rhs.w() - self.x() * rhs.x() - self.y() * rhs.y() - self.z() * rhs.z())
    
    def x(self): return self.q[0]
    def y(self): return self.q[1]
    def z(self): return self.q[2]
    def w(self): return self.q[3]
    def xyz(self): return self.q[:3]
    def xyzw(self): return self.q[:]

    def conj(self):
        return Quat(*(-self.xyz()), self.w())
    
    def dot(self, rhs):
        return np.sum(self.q * rhs.q)
    
    def normalize(self):
        return Quat(*(self.q / self.length()))
    
    def length(self):
        return np.linalg.norm(self.q)
    
    def length_squared(self):
        return np.sum(self.q**2)
    
    def rotate(self, p):
        x = (self.w() * p.x() + self.y() * p.z()) - (self.z() * p.y());
        y = (self.w() * p.y() + self.z() * p.x()) - (self.x() * p.z());
        z = (self.w() * p.z() + self.x() * p.y()) - (self.y() * p.x());
        w = (self.x() * p.x() + self.y() * p.y()) - (self.z() * p.z());
        return Float3(
            ((w * self.x() + x * self.w()) - y * self.z()) + z * self.y(),
            ((w * self.y() + y * self.w()) - z * self.x()) + x * self.z(),
            ((w * self.z() + z * self.w()) - x * self.y()) + y * self.x())

    @classmethod
    def unit(cls):
        return cls(0., 0., 0., 1.)
    
    @classmethod
    def zero(cls):
        return cls(0., 0., 0., 0.)
    
    @classmethod
    def rotation(cls, v, rad):
        angle = rad * 0.5
        s = math.sin(angle)
        c = math.cos(angle)
        return cls(*v.mul(s).xyz(), c)
    
    @classmethod
    def rotation_x(cls, rad):
        angle = rad * 0.5
        s = math.sin(angle)
        c = math.cos(angle)
        return cls(s, 0., 0., c)

    @classmethod
    def rotation_y(cls, rad):
        angle = rad * 0.5
        s = math.sin(angle)
        c = math.cos(angle)
        return cls(0., s, 0., c)
    
    @classmethod
    def rotation_z(cls, rad):
        angle = rad * 0.5
        s = math.sin(angle)
        c = math.cos(angle)
        return cls(0., 0., s, c)


def normalize(v):
    # return v / np.sqrt((v**2).sum())
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def angle(v1, v2):
    return math.acos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def saturate(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    return x

def clamp(x,a,b):
    if x < a:
        return a
    elif x > b:
        return b
    return x
    
def recip(x):
    return 1/x

def mix(a, b, t):
    return a*(1-t) + b*t

def step(edge, x):
    if x < edge:
        return 0
    else:
        return 1

def smoothstep(a, b, t):
    if a >= b:
        return 0
    else:
        x = saturate((t-a)/(b-a))
        return x*x*(3-2*t)

def sign(x):
    if x < 0.0:
        return -1
    else:
        return 1

def zerocheck(x):
    return max(x, np.finfo(float).eps)

def safe_pow(x,y):
    return pow(zerocheck(x), y)