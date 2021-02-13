import vecmath as vm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
from concurrent import futures
# from logging import StreamHandler, Formatter, INFO, getLogger

# def init_logger():
#     handler = StreamHandler()
#     handler.setLevel(INFO)
#     handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
#     logger = getLogger()
#     logger.addHandler(handler)
#     logger.setLevel(INFO)


class Ray:
    __slots__ = ('origin', 'direction')

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    
    def at(self, t: float):
        return self.origin + self.direction.mul(t)
    
    def clone(self):
        return Ray(self.origin.clone(), self.direction.clone())
    

class Camera:
    __slots__ = ('origin', 'uvw')

    def __init__(self, o, u, v, w):
        self.origin = o
        self.uvw = (u, v, w)
    
    @classmethod
    def lookat(cls, lookfrom, at, vup, vfov, aspect):
        u, v, w = (vm.Vec3(), vm.Vec3(), vm.Vec3())
        halfH = math.tan(math.radians(vfov)*0.5)
        halfW = aspect * halfH
        w = (lookfrom - at).normalize()
        u = vup.cross(w).normalize()
        v = w.cross(u)
        return cls(
            lookfrom.clone(),
            u.mul(2. * halfW),
            v.mul(2. * halfH),
            lookfrom - u.mul(halfW) - v.mul(halfH) - w)
    
    def get_ray(self, u: float, v: float):
        return Ray(
            self.origin.clone(),
            self.uvw[2] + self.uvw[0].mul(u) + self.uvw[1].mul(v) - self.origin)


class ONB:
    __slots__ = ('axis')

    def __init__(self, n):
        self.axis = [vm.Vec3(), vm.Vec3(), vm.Vec3()]
        self.axis[2] = n.normalize()
        if abs(self.axis[2].x()) > 0.9:
            self.axis[1] = self.axis[2].cross(vm.Vec3.yaxis()).normalize()
        else:
            self.axis[1] = self.axis[2].cross(vm.Vec3.xaxis()).normalize()
        self.axis[0] = self.axis[2].cross(self.axis[1])
    
    def u(self): return self.axis[0]
    def v(self): return self.axis[1]
    def w(self): return self.axis[2]
    
    def local(self, a):
        return self.axis[0].mul(a.x()) + self.axis[1].mul(a.y()) + self.axis[2].mul(a.z())


class Pdf:
    def value(self, rec, dir):
        raise NotImplementedErorr()
    
    def generate(self, rec):
        raise NotImplementedErorr()


class CosinePdf(Pdf):
    __slots__ = ()

    def value(self, rec, direction):
        cosine = direction.normalize().dot(rec.n)
        if cosine > 0.:
            return cosine / math.pi
        else:
            return 0.

    def generate(self, rec):
        return ONB(rec.n).local(vm.Vec3.random_cosine_direction())


class ShapePdf(Pdf):
    __slots__ = ('shape', 'origin')

    def __init__(self, shape, origin):
        self.shape = shape
        self.origin = origin
    
    def value(self, rec, direction):
        return self.shape.pdf_value(self.origin, direction)
    
    def generate(self, rec):
        return self.shape.random(self.origin)


class MixturePdf(Pdf):
    __slots__ = ('pdfs')

    def __init__(self, pdf0, pdf1):
        self.pdfs = [pdf0, pdf1]
    
    def value(self, rec, direction):
        pdf0_value = self.pdfs[0].value(rec, direction)
        pdf1_value = self.pdfs[1].value(rec, direction)
        return 0.5 * pdf0_value + 0.5 * pdf1_value
    
    def generate(self, rec):
        if vm.Float3.random().x() < 0.5:
            return self.pdfs[0].generate(rec)
        else:
            return self.pdfs[1].generate(rec)


class Scatter:
    __slots__ = ('ray', 'albedo', 'pdf', 'is_specular')

    def __init__(self):
        self.ray = Ray(vm.Point3(), vm.Vec3.xaxis())
        self.albedo = vm.Color()
        self.pdf = None
        self.is_specular = False


class Texture:
    def sample(self, u, v, p):
        raise NotImplementedErorr()


class ColorTexture(Texture):
    __slots__ = ('albedo')

    def __init__(self, albedo):
        self.albedo = albedo
    
    def sample(self, u, v, p):
        return self.albedo
    
    def clone(self):
        return ColorTexture(self.albedo)


class CheckerTexture(Texture):
    __slots__ = ('odd', 'even', 'freq')

    def __init__(self, odd, even, freq):
        self.odd = odd
        self.even = even
        self.freq = freq
    
    def sample(self, u, v, p):
        sine = math.sin(self.freq * p.x()) * math.sin(self.freq * p.y()) * math.sin(self.freq * p.z())
        if sine < 0.:
            return self.odd.sample(u, v, p)
        else:
            return self.even.sample(u, v, p)
    
    def clone(self):
        return CheckerTexture(self.odd, self.even, self.freq)


class ImageTexture(Texture):
    __slots__ = ('albedo', 'width', 'height')

    def __init__(self):
        self.albedo = None
        self.width = 0
        self.height = 0
    
    @classmethod
    def load(cls, path):
        image = cls()
        image.albedo = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(float) / 255.
        image.height, image.width, _ = image.albedo.shape
        return image
    
    def sample(self, u, v, p):
        x = int(u * float(self.width))
        y = int((1. - v) * float(self.height))
        u = 0 if x < 0 else self.width - 1 if x >= self.width else x
        v = 0 if y < 0 else self.height - 1 if y >= self.height else y
        return vm.Color(*self.albedo[v, u])
    
    def clone(self):
        image = ImageTexture()
        image.albedo = self.albedo
        image.width = self.width
        image.height = self.height
        return image


class Material:
    def scatter(self, ray, rec, srec):
        raise NotImplementedErorr()
    
    def emitted(self, ray, rec):
        return vm.Color()
    
    def scattering_pdf(self, ray, rec):
        return 0.


class Lambertian(Material):
    __slots__ = ('albedo')

    def __init__(self, albedo):
        self.albedo = albedo
    
    def scatter(self, ray, rec, srec):
        onb = ONB(rec.n)
        direction = onb.local(vm.Vec3.random_cosine_direction())
        srec.ray = Ray(rec.p.clone(), direction.normalize())
        srec.albedo = self.albedo.sample(rec.u, rec.v, rec.p)
        srec.pdf = CosinePdf()
        srec.is_specular = False
        return True
    
    def scattering_pdf(self, ray, rec):
        return max(ray.direction.normalize().dot(rec.n), 0.) / math.pi
    
    def clone(self):
        return Lambertian(self.albedo.clone())


class Metal(Material):
    __slots__ = ('albedo', 'fuzz')

    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = fuzz
    
    def scatter(self, ray, rec, srec):
        reflected = ray.direction.normalize().reflect(rec.n)
        reflected = reflected + vm.Vec3.random_in_unit_sphere().mul(self.fuzz)
        srec.ray = Ray(rec.p.clone(), reflected)
        srec.albedo = self.albedo.sample(rec.u, rec.v, rec.p)
        srec.is_specular = True
        return srec.ray.direction.dot(rec.n) > 0.
    
    def clone(self):
        return Metal(self.albedo.clone(), self.fuzz)


class Dielectric(Material):
    __slots__ = ('ri')

    def __init__(self, ri):
        self.ri = ri
    
    def schlick(self, cosine, ri):
        r0 = ((1. - ri) / (1. + ri))**2.
        return r0 + (1. - r0) * ((1. - cosine)**5.)
    
    def scatter(self, ray, rec, srec):
        reflected = ray.direction.normalize().reflect(rec.n)
        ni_over_nt = self.ri
        outward_normal = -rec.n
        cosine = self.ri * ray.direction.dot(rec.n) / ray.direction.length()
        if ray.direction.dot(rec.n) < 0.:
            outward_normal = rec.n.clone()
            ni_over_nt = 1. / self.ri
            cosine = - (ray.direction.dot(rec.n) / ray.direction.length())
        
        srec.albedo = vm.Color.one()
        srec.is_specular = True
        reflect_prob = 1.

        refracted = (-ray.direction).refract(outward_normal, ni_over_nt)
        if refracted:
            reflect_prob = self.schlick(cosine, self.ri)
        
        if vm.Float3.random().x() < reflect_prob:
            srec.ray = Ray(rec.p.clone(), reflected)
        else:
            srec.ray = Ray(rec.p.clone(), refracted)
        
        return True

    def clone(self):
        return Dielectric(self.ri)


class DiffuseLight(Material):
    __slots__ = ('emit')

    def __init__(self, emit):
        self.emit = emit
    
    def scatter(self, ray, rec, srec):
        return False
    
    def emitted(self, ray, rec):
        if ray.direction.dot(rec.n) < 0.:
            return self.emit.sample(rec.u, rec.v, rec.p)
        else:
            return vm.Color()
    
    def clone(self):
        return DiffuseLight(self.emit)


class HitRecord:
    __slots__ = ('t', 'p', 'n', 'm', 'u', 'v')

    def __init__(self):
        self.t = 0.
        self.p = vm.Point3()
        self.n = vm.Vec3.xaxis()
        self.u = 0.
        self.v = 0.
    
    def copy(self, other):
        self.t = other.t
        self.p = other.p.clone()
        self.n = other.n.clone()
        self.m = other.m.clone()
        self.u = other.u
        self.v = other.v


class Shape:
    def hit(self, ray, t0, t1, rec):
        raise NotImplementedErorr()
    
    def pdf_value(self, o, v):
        return 0.
    
    def random(self, o):
        return vm.Vec3.xaxis()


class Sphere(Shape):
    __slots__ = ('center', 'radius', 'material')

    def __init__(self, c, r, m):
        self.center = c
        self.radius = r
        self.material = m
    
    def hit(self, ray, t0, t1, rec):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2. * ray.direction.dot(oc)
        c = oc.dot(oc) - self.radius**2.
        d = b*b-4*a*c
        if d > 0:
            root = math.sqrt(d)
            temp = (-b - root) / (2. * a)
            if t0 < temp < t1:
                rec.t = temp
                rec.p = ray.at(temp)
                rec.n = (rec.p - self.center).div(self.radius)
                rec.m = self.material
                rec.u, rec.v = Sphere.get_sphere_uv(rec.n)
                return True
            temp = (-b + root) / (2. * a)
            if t0 < temp < t1:
                rec.t = temp
                rec.p = ray.at(temp)
                rec.n = (rec.p - self.center).div(self.radius)
                rec.m = self.material
                rec.u, rec.v = Sphere.get_sphere_uv(rec.n)
                return True
        
        return False
    
    def pdf_value(self, o, v):
        rec = HitRecord()
        if self.hit(Ray(o,v), 0.001, sys.float_info.max, rec):
            dd = (self.center - o).length_squared()
            rr = min(self.radius**2., dd)
            cos_theta_max = math.sqrt(1. - rr * (1. / dd))
            solid_angle = math.pi * 2. * (1. - cos_theta_max)
            return 1. / solid_angle
        else:
            return 0.
    
    def random(self, o):
        direction = self.center - o
        distance_squared = direction.length_squared()
        return ONB(direction).local(vm.Point3.random_to_sphere(self.radius, distance_squared))
    
    @classmethod
    def get_sphere_uv(cls, p):
        phi = math.atan2(p.z(), p.x())
        theta = math.asin(p.y())
        return (1. - (phi + math.pi) / (2. * math.pi), (theta + math.pi / 2.) / math.pi)


RECT_AXIS_TYPE_XY = 0
RECT_AXIS_TYPE_XZ = 1
RECT_AXIS_TYPE_YZ = 2


class Rect(Shape):
    __slots__ = ('x0', 'x1', 'y0', 'y1', 'k', 'axis', 'material')

    def __init__(self, x0, x1, y0, y1, k, axis, material):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.k = k
        self.axis = axis
        self.material = material
    
    def hit(self, ray, t0, t1, rec):
        org = ray.origin.clone()
        dir = ray.direction.clone()
        axis = vm.Vec3.zaxis()
        if self.axis == RECT_AXIS_TYPE_XZ:
            org = vm.Point3(org.x(), org.z(), org.y())
            dir = vm.Vec3(dir.x(), dir.z(), dir.y())
            axis = vm.Vec3.yaxis()
        elif self.axis == RECT_AXIS_TYPE_YZ:
            org = vm.Point3(org.y(), org.z(), org.x())
            dir = vm.Vec3(dir.y(), dir.z(), dir.x())
            axis = vm.Vec3.xaxis()
        
        t = (self.k - org.z()) / dir.z()
        if t < t0 or t > t1:
            return False
        
        x = org.x() + t * dir.x()
        y = org.y() + t * dir.y()
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return False
        
        rec.u = (x - self.x0) / (self.x1 - self.x0)
        rec.v = (y - self.y0) / (self.y1 - self.y0)
        rec.t = t
        rec.m = self.material.clone()
        rec.p = ray.at(t)
        rec.n = axis
        return True
    
    def pdf_value(self, o, v):
        if self.axis != RECT_AXIS_TYPE_XZ:
            return 0.
        
        rec = HitRecord()
        if self.hit(Ray(o,v), 0.001, sys.float_info.max, rec):
            area = (self.x1 - self.x0) * (self.y1 - self.y0)
            distance_squared = rec.t**2. * v.length_squared()
            cosine = abs(v.dot(rec.n)) / v.length()
            return distance_squared / (cosine * area)
        else:
            return 0.
    
    def random(self, o):
        if self.axis != RECT_AXIS_TYPE_XZ:
            return vm.Vec3.xaxis()
        
        rnd = vm.Vec3.random()
        x = self.x0 + rnd.x() * (self.x1 - self.x0)
        y = self.y0 + rnd.y() * (self.y1 - self.y0)
        random_point = vm.Point3(x, y, self.k)
        if self.axis == RECT_AXIS_TYPE_XZ:
            random_point = vm.Point3(x, self.k, y)
        elif self.axis == RECT_AXIS_TYPE_YZ:
            random_point = vm.Point3(self.k, x, y)
        return random_point - o


class Box(Shape):
    __slots__ = ('shapes', 'p0', 'p1')

    def __init__(self, p0, p1, mat):
        self.p0 = p0
        self.p1 = p1
        self.shapes = ShapeList()
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_xy(p0.x(), p1.x(), p0.y(), p1.y(), p1.z())
            .build())
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_xy(p0.x(), p1.x(), p0.y(), p1.y(), p0.z())
            .flip_face()
            .build())
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_xz(p0.x(), p1.x(), p0.z(), p1.z(), p1.y())
            .flip_face()
            .build())
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_xz(p0.x(), p1.x(), p0.z(), p1.z(), p0.y())
            .build())
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_yz(p0.y(), p1.y(), p0.z(), p1.z(), p1.x())
            .build())
        self.shapes.add(ShapeBuilder()
            .set_material(mat)
            .rect_yz(p0.y(), p1.y(), p0.z(), p1.z(), p0.x())
            .flip_face()
            .build())
    
    def hit(self, ray, t0, t1, rec):
        return self.shapes.hit(ray, t0, t1, rec)



class FlipFace(Shape):
    __slots__ = ('shape')

    def __init__(self, shape):
        self.shape = shape
    
    def hit(self, ray, t0, t1, rec):
        if self.shape.hit(ray, t0, t1, rec):
            rec.n = -rec.n
            return True
        else:
            return False


class Translate(Shape):
    __slots__ = ('shape', 'offset')

    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset

    def hit(self, ray, t0, t1, rec):
        move_ray = Ray(ray.origin - self.offset, ray.direction.clone())
        if self.shape.hit(move_ray, t0, t1, rec):
            rec.p = rec.p + self.offset
            return True
        else:
            return False


class Rotate(Shape):
    __slots__ = ('shape', 'rot')

    def __init__(self, shape, axis, angle):
        self.shape = shape
        self.rot = vm.Quat.rotation(axis, math.radians(angle))
    
    def hit(self, ray, t0, t1, rec):
        rev_rot = self.rot.conj()
        origin = rev_rot.rotate(ray.origin)
        direction = rev_rot.rotate(ray.direction)
        rot_ray = Ray(origin, direction)
        if self.shape.hit(rot_ray, t0, t1, rec):
            rec.p = self.rot.rotate(rec.p)
            rec.n = self.rot.rotate(rec.n)
            return True
        else:
            return False


class ShapeList(Shape):
    __slots__ = ('shapes')

    def __init__(self):
        self.shapes = []
    
    def add(self, shape):
        self.shapes.append(shape)

    def hit(self, ray, t0, t1, rec):
        tmp_rec = HitRecord()
        hit_anything = False
        closest_so_far = t1
        for shape in self.shapes:
            if shape.hit(ray, t0, closest_so_far, tmp_rec):
                hit_anything = True
                closest_so_far = tmp_rec.t
                rec.copy(tmp_rec)
        return hit_anything
    
    def pdf_value(self, o, v):
        weight = 1. / float(len(self.shapes))
        return sum(weight * s.pdf_value(o,v) for s in self.shapes)
    
    def random(self, o):
        n = len(self.shapes)
        if n == 0:
            return vm.Point3()
        index = int(float(n) * vm.Float3.random().x())
        if index > 0 and index >= n:
            index = n - 1
        return self.shapes[index].random(o)
        


class ShapeBuilder:
    __slots__ = ('texture', 'material', 'shape')

    def __init__(self):
        self.texture = None
        self.material = None
        self.shape = None
    
    def set_texture(self, tex):
        self.texture = tex
        return self
    
    def color_texture(self, albedo):
        self.texture = ColorTexture(albedo)
        return self
    
    def checker_texture(self, odd, even, freq):
        self.texture = CheckerTexture(ColorTexture(odd), ColorTexture(even), freq)
        return self
    
    def image_texture(self, path):
        self.texture = ImageTexture.load(path)
        return self
    
    def set_material(self, mat):
        self.material = mat
        return self

    def lambertian(self):
        self.material = Lambertian(self.texture)
        return self
    
    def metal(self, fuzz):
        self.material = Metal(self.texture, fuzz)
        return self
    
    def dielectric(self, ri):
        self.material = Dielectric(ri)
        return self
    
    def diffuse_light(self):
        self.material = DiffuseLight(self.texture)
        return self
    
    def default_material(self):
        self.material = Lambertian(ColorTexture(vm.Color.one()))
        return self
    
    def sphere(self, c, r):
        self.shape = Sphere(c, r, self.material)
        return self

    def rect_xy(self, x0, x1, y0, y1, k):
        self.shape = Rect(x0, x1, y0, y1, k, RECT_AXIS_TYPE_XY, self.material)
        return self
    
    def rect_xz(self, x0, x1, y0, y1, k):
        self.shape = Rect(x0, x1, y0, y1, k, RECT_AXIS_TYPE_XZ, self.material)
        return self
    
    def rect_yz(self, x0, x1, y0, y1, k):
        self.shape = Rect(x0, x1, y0, y1, k, RECT_AXIS_TYPE_YZ, self.material)
        return self
    
    def box(self, p0, p1):
        self.shape = Box(p0, p1, self.material)
        return self
    
    def flip_face(self):
        self.shape = FlipFace(self.shape)
        return self
    
    def translate(self, offset):
        self.shape = Translate(self.shape, offset)
        return self
    
    def rotate(self, axis, angle):
        self.shape = Rotate(self.shape, axis, angle)
        return self
    
    def build(self):
        return self.shape


class Scene:
    __slots__ = ('samples', 'depth', 'bgcolor', 'image', 'camera', 'world', 'light')

    def __init__(self, width, height, samples, depth):
        self.samples = samples
        self.depth = depth
        self.bgcolor = vm.Color.full(0.2)
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
    
    def build(self):
        self.camera = Camera(
            vm.Point3(),
            vm.Vec3(4., 0., 0.),
            vm.Vec3(0., 2., 0.),
            vm.Vec3(-2., -1., -1.))
        
        self.world = ShapeList()
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color(0.1, 0.2, 0.5))
            .lambertian()
            .sphere(vm.Point3(0.6, 0, -1.), 0.5)
            .build())
        # self.world.add(ShapeBuilder()
        #     .image_texture('assets/brick_diffuse.jpg')
        #     .lambertian()
        #     .sphere(vm.Point3(0.6, 0, -1.), 0.5)
        #     .build())
        # self.world.add(ShapeBuilder()
        #     .color_texture(vm.Color.full(10.0))
        #     .diffuse_light()
        #     .sphere(vm.Point3(-0.6, 0, -1.), 0.5)
        #     .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.8))
            .metal(0.2)
            .sphere(vm.Point3(-0.6, 0, -1.), 0.5)
            .build())
        # self.world.add(ShapeBuilder()
        #     .dielectric(1.5)
        #     .sphere(vm.Point3(-0.6, 0, -1.), 0.5)
        #     .build())
        # self.world.add(ShapeBuilder()
        #     .dielectric(1.5)
        #     .sphere(vm.Point3(-0.6, 0, -1.), -0.45)
        #     .build())
        # self.world.add(ShapeBuilder()
        #     .color_texture(vm.Color.full(0.8))
        #     .metal(0.2)
        #     .sphere(vm.Point3(0., -0.35, -0.8), 0.15)
        #     .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color(10., 0., 0.))
            .diffuse_light()
            .rect_xy(-2., 2., 0.2, 2., -2.)
            .build())
        self.world.add(ShapeBuilder()
            .checker_texture(vm.Color(0.8, 0.8, 0.), vm.Color(0.8, 0.2, 0.), 10.)
            .lambertian()
            .sphere(vm.Point3(0., -100.5, -1.), 100.)
            .build())
    
    def build_random(self):
        ny, nx, _ = self.image.shape
        self.camera = Camera.lookat(
            vm.Point3(13., 2., 3.),
            vm.Point3(),
            vm.Vec3.yaxis(),
            20.,
            float(nx) / float(ny))
        
        self.world = ShapeList()
        N = 11
        for i in range(-N,N):
            for j in range(-N,N):
                rnd = vm.Float3.random()
                choose_mat = rnd.x()
                ofx = rnd.y()
                ofz = rnd.z()
                center = vm.Point3(float(i) + 0.9 * ofx, 0.2, float(j)+0.9*ofz)
                if (center - vm.Point3(4., 0.2, 0.)).length() > 0.9:
                    if choose_mat < 0.8:
                        self.world.add(ShapeBuilder()
                            .color_texture(vm.Color.random())
                            .lambertian()
                            .sphere(center, 0.2)
                            .build())
                    elif choose_mat < 0.95:
                        self.world.add(ShapeBuilder()
                            .color_texture(vm.Color.random())
                            .metal(0.5*rnd.x())
                            .sphere(center, 0.2)
                            .build())
                    else:
                        self.world.add(ShapeBuilder()
                            .dielectric(1.5)
                            .sphere(center, 0.2)
                            .build())
        
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.5))
            .lambertian()
            .sphere(vm.Point3(0., -1000., -1.), 1000.)
            .build())
    
    def build_cornell_box(self):
        ny, nx, _ = self.image.shape
        self.camera = Camera.lookat(
            vm.Point3(278., 278., -800.),
            vm.Point3(278., 278., 0.),
            vm.Vec3.yaxis(),
            40.,
            float(nx) / float(ny))
        
        self.world = ShapeList()
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color(0.12, 0.45, 0.15))
            .lambertian()
            .rect_yz(0., 555., 0., 555., 555.)
            .flip_face()
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color(0.65, 0.05, 0.05))
            .lambertian()
            .rect_yz(0., 555., 0., 555., 0.)
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(15.))
            .diffuse_light()
            .rect_xz(213., 343., 227., 332., 554.)
            .flip_face()
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.73))
            .lambertian()
            .rect_xz(0., 555., 0., 555., 555.)
            .flip_face()
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.73))
            .lambertian()
            .rect_xz(0., 555., 0., 555., 0.)
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.73))
            .lambertian()
            .rect_xy(0., 555., 0., 555., 555.)
            .flip_face()
            .build())
        
        self.world.add(ShapeBuilder()
            .dielectric(1.5)
            .sphere(vm.Point3(190., 90., 190.), 90.)
            .build())
        self.world.add(ShapeBuilder()
            .color_texture(vm.Color.full(0.73))
            .lambertian()
            .box(vm.Point3(), vm.Point3(165., 330., 165.))
            .rotate(vm.Vec3.yaxis(), 15.)
            .translate(vm.Point3(265., 0., 295.))
            .build())

        self.light = ShapeList()
        self.light.add(ShapeBuilder()
            .default_material()
            .rect_xz(213., 343., 227., 332., 554.)
            .build())
        self.light.add(ShapeBuilder()
            .default_material()
            .sphere(vm.Point3(190., 90., 190.), 90.)
            .build())
        
        self.bgcolor = vm.Color()
    
    def render(self):
        ny, nx, _ = self.image.shape
        for j in range(ny):
            for i in range(nx):
                c = vm.Color()
                for s in range(self.samples):
                    rnd = vm.Float3.random()
                    u = (float(i)+rnd.x()) / float(nx)
                    v = (float(j)+rnd.y()) / float(ny)
                    r = self.camera.get_ray(u, v)
                    c = c + self.color(rm, 0)
                self.image[ny-j-1,i] = c.div(float(self.samples)).saturate().gamma().as_bgr8()
    
    def render_line(self, y):
        ny, nx, _ = self.image.shape
        image = np.zeros((nx, 3), dtype=np.uint8)
        for i in range(nx):
            c = vm.Color()
            for s in range(self.samples):
                rnd = vm.Float3.random()
                u = (float(i)+rnd.x()) / float(nx)
                v = (float(y)+rnd.y()) / float(ny)
                r = self.camera.get_ray(u, v)
                c = c + self.color(r, 0)
            image[i] = c.div(float(self.samples)).saturate().gamma().as_bgr8()
        return image

    def color(self, ray, depth):
        rec = HitRecord()
        if self.world.hit(ray, 0.001, sys.float_info.max, rec):
            emit = rec.m.emitted(ray, rec)
            srec = Scatter()
            if depth < self.depth and rec.m.scatter(ray, rec, srec):
                if srec.is_specular:
                    return emit + (srec.albedo * self.color(srec.ray, depth+1))
                
                spdf = ShapePdf(self.light, rec.p.clone())
                mixpdf = MixturePdf(spdf, srec.pdf)
                srec.ray = Ray(rec.p.clone(), mixpdf.generate(rec))
                pdf_value = mixpdf.value(rec, srec.ray.direction)
                if pdf_value > 0.:
                    spdf_value = rec.m.scattering_pdf(srec.ray, rec)
                    albedo = srec.albedo.mul(spdf_value)
                    return emit + (albedo * self.color(srec.ray, depth+1)).div(pdf_value)
                else:
                    return emit
            else:
                return emit
        return self.background(ray.direction)
    
    def background(self, d):
        return self.bgcolor
    
    def backgroundSky(self, d):
        t = 0.5 * (d.normalize().y() + 1.)
        return vm.Color.one().lerp(vm.Color(0.5, 0.7, 1.), t)


class RenderBlock:
    __slots__ = ('scene', 'line')

    def __init__(self, scene, line):
        self.scene = scene
        self.line = line
    
    def render(self):
        return self.scene.render_line(self.line)


def render_block(block):
    return block.render()


if __name__ == "__main__":
    nx = 200
    ny = 200
    ns = 100
    max_depth = 50

    # init_logger()

    start = time.time()
    scene = Scene(nx, ny, ns, max_depth)
    # scene.build()
    # scene.build_random()
    scene.build_cornell_box()
    # scene.render()

    rows = []
    blocks = [RenderBlock(scene, line) for line in range(ny)]
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(render_block, blocks))
    image = np.stack(reversed(rows))
    cv2.imwrite('render.png', image)

    end = time.time() - start
    print("time: {:.2f}s ({:.2f}m).".format(end, end/60))
    
    # cv2.imshow("image", scene.image)
    cv2.imshow("image", image)
    while cv2.waitKey(33) != 27:
        pass
