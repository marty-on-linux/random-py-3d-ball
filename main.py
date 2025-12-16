"""
3D Game Engine with Silver Reflective Ball, Grass, Sun, and Shadows
Python 3.12 Compatible
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random
from scipy import ndimage
from scipy.spatial import Delaunay
from PIL import Image
import trimesh
import glm

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FOV = 60
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0

# Physics constants
GRAVITY = -9.81
FRICTION = 0.98
BOUNCE_DAMPING = 0.7
AIR_RESISTANCE = 0.999

# Sun settings
SUN_POSITION = [30.0, 45.0, 25.0]
SUN_COLOR = (1.0, 0.95, 0.8)

# Wind settings
WIND_STRENGTH = 0.3  # Light breeze
WIND_FREQUENCY = 0.8  # How fast wind oscillates
WIND_DIRECTION = (1.0, 0.0, 0.5)  # Normalized later

# Colors - hyperrealistic
SKYBOX_TOP = (0.25, 0.55, 0.95)
SKYBOX_HORIZON = (0.7, 0.85, 1.0)
SKYBOX_BOTTOM = (0.55, 0.75, 0.65)
GRASS_COLOR_DARK = (0.08, 0.28, 0.05)
GRASS_COLOR_LIGHT = (0.2, 0.5, 0.12)


class Vector3:
    """3D Vector class for physics calculations"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def to_tuple(self):
        return (self.x, self.y, self.z)


class Camera:
    """Realistic first-person camera with smooth movement"""
    def __init__(self):
        self.position = Vector3(0.0, 2.0, 8.0)
        self.velocity = Vector3()
        self.yaw = -90.0
        self.pitch = -10.0
        self.front = Vector3(0.0, 0.0, -1.0)
        self.up = Vector3(0.0, 1.0, 0.0)
        self.right = Vector3(1.0, 0.0, 0.0)
        self.world_up = Vector3(0.0, 1.0, 0.0)
        
        self.move_speed = 5.0
        self.mouse_sensitivity = 0.1
        self.smoothing = 0.15
        self.target_yaw = self.yaw
        self.target_pitch = self.pitch
        
        self.bob_time = 0.0
        self.bob_amount = 0.03
        self.bob_speed = 8.0
        self.is_moving = False
        
        self.update_vectors()
    
    def update_vectors(self):
        """Update camera direction vectors"""
        front = Vector3()
        front.x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front.y = math.sin(math.radians(self.pitch))
        front.z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = front.normalize()
        
        self.right = self.front.cross(self.world_up).normalize()
        self.up = self.right.cross(self.front).normalize()
    
    def process_mouse(self, xoffset, yoffset):
        """Handle mouse input for camera rotation"""
        self.target_yaw += xoffset * self.mouse_sensitivity
        self.target_pitch += yoffset * self.mouse_sensitivity
        
        # Clamp pitch
        self.target_pitch = max(-89.0, min(89.0, self.target_pitch))
    
    def update(self, delta_time, keys):
        """Update camera position and rotation"""
        # Smooth rotation interpolation
        self.yaw += (self.target_yaw - self.yaw) * self.smoothing
        self.pitch += (self.target_pitch - self.pitch) * self.smoothing
        self.update_vectors()
        
        # Movement
        move_dir = Vector3()
        self.is_moving = False
        
        if keys[K_w]:
            move_dir = move_dir + Vector3(self.front.x, 0, self.front.z).normalize()
            self.is_moving = True
        if keys[K_s]:
            move_dir = move_dir - Vector3(self.front.x, 0, self.front.z).normalize()
            self.is_moving = True
        if keys[K_a]:
            move_dir = move_dir - self.right
            self.is_moving = True
        if keys[K_d]:
            move_dir = move_dir + self.right
            self.is_moving = True
        if keys[K_SPACE]:
            move_dir = move_dir + self.world_up
            self.is_moving = True
        if keys[K_LSHIFT]:
            move_dir = move_dir - self.world_up
            self.is_moving = True
        
        # Apply movement with acceleration
        if move_dir.magnitude() > 0:
            move_dir = move_dir.normalize()
            self.velocity = self.velocity + move_dir * self.move_speed * delta_time * 5
        
        # Apply friction
        self.velocity = self.velocity * (1.0 - 3.0 * delta_time)
        
        # Clamp velocity
        max_speed = self.move_speed
        if self.velocity.magnitude() > max_speed:
            self.velocity = self.velocity.normalize() * max_speed
        
        # Update position
        self.position = self.position + self.velocity * delta_time
        
        # Head bob
        if self.is_moving and abs(self.velocity.magnitude()) > 0.1:
            self.bob_time += delta_time * self.bob_speed
        else:
            self.bob_time = 0
    
    def get_view_matrix_params(self):
        """Get parameters for gluLookAt"""
        bob_offset = math.sin(self.bob_time) * self.bob_amount if self.is_moving else 0
        pos = self.position
        target = pos + self.front
        return (
            pos.x, pos.y + bob_offset, pos.z,
            target.x, target.y + bob_offset, target.z,
            self.up.x, self.up.y, self.up.z
        )


class PhysicsObject:
    """Base class for physics-enabled objects"""
    def __init__(self, position, mass=1.0):
        self.position = position
        self.velocity = Vector3()
        self.acceleration = Vector3()
        self.mass = mass
        self.is_grounded = False
    
    def apply_force(self, force):
        self.acceleration = self.acceleration + force * (1.0 / self.mass)
    
    def update(self, delta_time):
        # Apply gravity
        gravity_force = Vector3(0, GRAVITY * self.mass, 0)
        self.apply_force(gravity_force)
        
        # Update velocity
        self.velocity = self.velocity + self.acceleration * delta_time
        
        # Apply air resistance
        self.velocity = self.velocity * AIR_RESISTANCE
        
        # Update position
        self.position = self.position + self.velocity * delta_time
        
        # Reset acceleration
        self.acceleration = Vector3()


class ReflectiveBall(PhysicsObject):
    """
    Physically-based chrome silver reflective ball
    Uses Fresnel equations and metallic BRDF approximation
    """
    def __init__(self, position, radius=0.5):
        super().__init__(position, mass=1.0)
        self.radius = radius
        self.rotation = Vector3()
        self.angular_velocity = Vector3()
        self.quadric = None  # Reuse quadric for performance
        
        # Physically-based chrome material
        # Chrome has ~65% reflectivity at normal incidence (F0)
        # Using metallic workflow approximation for OpenGL fixed function
        
        # Fresnel F0 for chrome ≈ 0.55-0.65
        f0 = 0.60
        
        # Ambient: low, represents indirect illumination on metal
        self.ambient = [f0 * 0.3, f0 * 0.3, f0 * 0.32, 1.0]
        
        # Diffuse: very low for metals (most light is reflected specularly)
        self.diffuse = [f0 * 0.15, f0 * 0.15, f0 * 0.16, 1.0]
        
        # Specular: high, represents mirror-like reflection
        self.specular = [0.98, 0.98, 0.99, 1.0]
        
        # Shininess: very high for mirror-like surface
        # Roughness α ≈ 0.05 maps to shininess ≈ 2/α² = 800
        self.shininess = 128.0  # OpenGL max is typically 128
        
        # Slight emission to simulate environment reflection boost
        self.emission = [0.08, 0.08, 0.09, 1.0]
    
    def check_ground_collision(self, ground_y=0.0):
        """Check collision with ground using coefficient of restitution"""
        if self.position.y - self.radius < ground_y:
            self.position.y = ground_y + self.radius
            if self.velocity.y < 0:
                # Coefficient of restitution for chrome on grass ≈ 0.6-0.7
                self.velocity.y = -self.velocity.y * BOUNCE_DAMPING
                
                # Angular velocity from friction torque
                # τ = r × F, where F is friction force
                self.angular_velocity = Vector3(
                    self.velocity.z * 3,
                    0,
                    -self.velocity.x * 3
                )
                self.is_grounded = True
    
    def update(self, delta_time):
        super().update(delta_time)
        
        if self.is_grounded:
            self.velocity.x *= FRICTION
            self.velocity.z *= FRICTION
            self.angular_velocity = self.angular_velocity * 0.98
        
        self.rotation = self.rotation + self.angular_velocity * delta_time
        self.is_grounded = False
        self.check_ground_collision(0.0)
    
    def draw(self, env_texture):
        """
        Draw chrome ball with physically-based environment reflections
        Uses sphere mapping to simulate mirror reflection
        """
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(math.degrees(self.rotation.x), 1, 0, 0)
        glRotatef(math.degrees(self.rotation.y), 0, 1, 0)
        glRotatef(math.degrees(self.rotation.z), 0, 0, 1)
        
        # Set material properties
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, self.emission)
        
        # Environment mapping setup
        glColor4f(1, 1, 1, 1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, env_texture)
        
        # Use MODULATE to blend texture with lighting
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        # Sphere mapping for reflection
        glEnable(GL_TEXTURE_GEN_S)
        glEnable(GL_TEXTURE_GEN_T)
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
        
        # Create quadric once and reuse
        if not self.quadric:
            self.quadric = gluNewQuadric()
            gluQuadricNormals(self.quadric, GLU_SMOOTH)
            gluQuadricTexture(self.quadric, GL_TRUE)
        
        # High-poly sphere for smooth reflections
        gluSphere(self.quadric, self.radius, 64, 64)
        
        # Cleanup
        glDisable(GL_TEXTURE_GEN_S)
        glDisable(GL_TEXTURE_GEN_T)
        glDisable(GL_TEXTURE_2D)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0, 0, 0, 1])
        glPopMatrix()


class Wind:
    """Lightweight wind simulation"""
    def __init__(self):
        self.time = 0
        self.strength = WIND_STRENGTH
        self.frequency = WIND_FREQUENCY
        mag = math.sqrt(WIND_DIRECTION[0]**2 + WIND_DIRECTION[1]**2 + WIND_DIRECTION[2]**2)
        self.direction = (WIND_DIRECTION[0]/mag, WIND_DIRECTION[1]/mag, WIND_DIRECTION[2]/mag)
    
    def update(self, delta_time):
        self.time += delta_time
    
    def get_wind_force(self):
        """Get wind force for physics"""
        gust = math.sin(self.time * 0.7) * 0.3 + math.sin(self.time * 1.3) * 0.2
        force = (self.strength + gust) * 0.08
        return Vector3(self.direction[0] * force, 0, self.direction[2] * force)


class GrassGround:
    """
    High-quality grass ground with procedural texture and animated grass blades
    Uses Perlin-like noise for natural appearance and LOD for performance
    """
    def __init__(self, size=80, wind=None):
        self.size = size
        self.wind = wind
        self.grass_texture = None
        self.grass_data = []
        self.ground_list = None
        self.grass_list_near = None  # Display list for near grass
        self.create_grass_texture()
        self.generate_grass_data()
        self.create_ground_display_list()
    
    def _simplex_noise_2d(self, x, y):
        """
        Simplified 2D noise function using gradient noise approximation
        Based on improved Perlin noise but optimized for real-time use
        """
        # Skew to simplex grid
        F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        G2 = (3.0 - math.sqrt(3.0)) / 6.0
        
        s = (x + y) * F2
        i = math.floor(x + s)
        j = math.floor(y + s)
        
        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0
        
        # Determine simplex
        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1
        
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        
        # Hash gradient indices
        def grad(h, gx, gy):
            h = h & 7
            u = gx if h < 4 else gy
            v = gy if h < 4 else gx
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        
        # Permutation using simple hash
        def perm(x):
            return int((x * 1103515245 + 12345) & 0x7fffffff) % 256
        
        ii = int(i) & 255
        jj = int(j) & 255
        
        gi0 = perm(ii + perm(jj))
        gi1 = perm(ii + i1 + perm(jj + j1))
        gi2 = perm(ii + 1 + perm(jj + 1))
        
        # Calculate contribution from corners
        n0 = n1 = n2 = 0.0
        
        t0 = 0.5 - x0*x0 - y0*y0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * grad(gi0, x0, y0)
        
        t1 = 0.5 - x1*x1 - y1*y1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * grad(gi1, x1, y1)
        
        t2 = 0.5 - x2*x2 - y2*y2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * grad(gi2, x2, y2)
        
        # Scale to [0, 1]
        return (70.0 * (n0 + n1 + n2) + 1.0) * 0.5
    
    def create_grass_texture(self):
        """
        Create high-quality procedural grass texture using multi-octave noise
        Simulates grass clumps, dirt patches, and color variation
        """
        self.grass_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.grass_texture)
        
        size = 512  # Higher resolution for quality
        pixels = []
        
        for y in range(size):
            for x in range(size):
                nx = x / size * 4  # Scale for noise
                ny = y / size * 4
                
                # Multi-octave noise (fBm - fractal Brownian motion)
                # f(x) = Σ amplitude^i * noise(frequency^i * x)
                noise = 0
                amplitude = 1.0
                frequency = 1.0
                max_value = 0
                
                for octave in range(4):
                    noise += amplitude * self._simplex_noise_2d(nx * frequency, ny * frequency)
                    max_value += amplitude
                    amplitude *= 0.5  # Persistence
                    frequency *= 2.0  # Lacunarity
                
                noise /= max_value  # Normalize to [0, 1]
                
                # Secondary noise for detail
                detail = self._simplex_noise_2d(nx * 8, ny * 8) * 0.3
                
                # Color mapping
                # Base grass color with variation
                base_r = 0.06 + noise * 0.12 + detail * 0.02
                base_g = 0.22 + noise * 0.28 + detail * 0.05
                base_b = 0.03 + noise * 0.06 + detail * 0.01
                
                # Dirt patches (low noise = dirt)
                if noise < 0.25:
                    dirt_factor = (0.25 - noise) / 0.25
                    base_r = base_r * (1 - dirt_factor) + 0.15 * dirt_factor
                    base_g = base_g * (1 - dirt_factor) + 0.12 * dirt_factor
                    base_b = base_b * (1 - dirt_factor) + 0.08 * dirt_factor
                
                # Lush patches (high noise = healthy grass)
                if noise > 0.7:
                    lush_factor = (noise - 0.7) / 0.3
                    base_g += 0.1 * lush_factor
                
                # Clamp and convert
                r = int(max(0, min(255, base_r * 255)))
                g = int(max(0, min(255, base_g * 255)))
                b = int(max(0, min(255, base_b * 255)))
                
                pixels.extend([r, g, b])
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, bytes(pixels))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        # Try to enable anisotropic filtering
        try:
            from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0)
        except:
            pass
    
    def create_ground_display_list(self):
        """Pre-compile ground geometry with higher subdivision"""
        self.ground_list = glGenLists(1)
        glNewList(self.ground_list, GL_COMPILE)
        
        s = self.size / 2
        subdivisions = 8  # More subdivisions for better lighting
        step = self.size / subdivisions
        tex_repeat = 20
        tex_step = tex_repeat / subdivisions
        
        glBegin(GL_QUADS)
        for i in range(subdivisions):
            for j in range(subdivisions):
                x0, z0 = -s + i * step, -s + j * step
                x1, z1 = x0 + step, z0 + step
                t0, t1 = i * tex_step, (i + 1) * tex_step
                s0, s1 = j * tex_step, (j + 1) * tex_step
                
                glNormal3f(0, 1, 0)
                glTexCoord2f(t0, s0); glVertex3f(x0, 0, z0)
                glTexCoord2f(t0, s1); glVertex3f(x0, 0, z1)
                glTexCoord2f(t1, s1); glVertex3f(x1, 0, z1)
                glTexCoord2f(t1, s0); glVertex3f(x1, 0, z0)
        glEnd()
        glEndList()
    
    def generate_grass_data(self):
        """Generate grass blade data with natural clustering"""
        random.seed(123)
        
        # Use Poisson disk sampling approximation for natural distribution
        for _ in range(4000):
            # Clustered distribution
            if random.random() < 0.7:
                # Clustered grass
                cx = random.uniform(-self.size/2, self.size/2)
                cz = random.uniform(-self.size/2, self.size/2)
                for _ in range(3):
                    x = cx + random.gauss(0, 0.5)
                    z = cz + random.gauss(0, 0.5)
                    if abs(x) < self.size/2 and abs(z) < self.size/2:
                        height = random.uniform(0.15, 0.35)
                        angle = random.uniform(0, 360)
                        tilt = random.gauss(0, 5)
                        color_var = random.gauss(1.0, 0.12)
                        self.grass_data.append((x, z, height, angle, tilt, color_var))
            else:
                # Scattered grass
                x = random.uniform(-self.size/2, self.size/2)
                z = random.uniform(-self.size/2, self.size/2)
                height = random.uniform(0.1, 0.25)
                angle = random.uniform(0, 360)
                tilt = random.gauss(0, 8)
                color_var = random.gauss(1.0, 0.15)
                self.grass_data.append((x, z, height, angle, tilt, color_var))
    
    def draw(self, camera_pos=None):
        """Draw grass ground with proper material setup"""
        glDisable(GL_CULL_FACE)
        
        # Ground material
        glColor4f(1, 1, 1, 1)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.06, 0.12, 0.03, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.15, 0.35, 0.08, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.01, 0.02, 0.01, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 4.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0, 0, 0, 1])
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.grass_texture)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glCallList(self.ground_list)
        glDisable(GL_TEXTURE_2D)
        
        glEnable(GL_CULL_FACE)
        
        self.draw_grass_blades(camera_pos)
    
    def draw_grass_blades(self, camera_pos=None):
        """Draw grass blades with wind animation and LOD"""
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        wind_time = self.wind.time if self.wind else 0
        cam_x = camera_pos.x if camera_pos else 0
        cam_z = camera_pos.z if camera_pos else 0
        
        # Batch all grass into single begin/end
        glBegin(GL_TRIANGLES)
        for x, z, height, angle, tilt, color_var in self.grass_data:
            # Distance culling
            dx, dz = x - cam_x, z - cam_z
            dist_sq = dx*dx + dz*dz
            if dist_sq > 400:  # 20^2
                continue
            
            # Simple wind
            wind_tilt = math.sin(wind_time * 0.8 + x * 0.25 + z * 0.2) * 15
            total_tilt = tilt + wind_tilt
            
            # Pre-compute
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            bend = math.radians(total_tilt)
            tip_y = height * math.cos(bend)
            tip_off = height * math.sin(bend)
            
            # Colors
            br, bg, bb = 0.1 * color_var, 0.24 * color_var, 0.04 * color_var
            tr, tg, tb = 0.28 * color_var, 0.55 * color_var, 0.14 * color_var
            
            # Transform grass blade
            w = 0.018
            bx1, bz1 = x + cos_a * (-w), z + sin_a * (-w)
            bx2, bz2 = x + cos_a * w, z + sin_a * w
            tx, tz = x + cos_a * tip_off, z + sin_a * tip_off
            
            glColor3f(br, bg, bb)
            glVertex3f(bx1, 0.001, bz1)
            glVertex3f(bx2, 0.001, bz2)
            glColor3f(tr, tg, tb)
            glVertex3f(tx, tip_y, tz)
        glEnd()
        
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glColor4f(1, 1, 1, 1)


class Sun:
    """
    Physically-based sun rendering with limb darkening, corona, and god rays
    Uses atmospheric scattering approximation for realistic glow
    """
    def __init__(self):
        self.position = Vector3(*SUN_POSITION)
        self.color = SUN_COLOR
        self.size = 2.5
        self.time = 0
        self.quadric = None
        self.corona_list = None
        self._create_corona_geometry()
    
    def _create_corona_geometry(self):
        """Pre-compile corona geometry for performance"""
        self.corona_list = glGenLists(1)
        glNewList(self.corona_list, GL_COMPILE)
        
        # Corona as radial gradient triangles
        segments = 48
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)  # Center (colored at draw time)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(math.cos(angle), math.sin(angle), 0)
        glEnd()
        glEndList()
    
    def update(self, delta_time):
        self.time += delta_time
    
    def draw(self):
        """Draw sun with physically-based corona and limb darkening"""
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        if not self.quadric:
            self.quadric = gluNewQuadric()
        
        # Sun disk with limb darkening effect
        # Draw multiple layers to simulate limb darkening: I(r) = I0(1 - u(1 - √(1-r²)))
        # Center is brightest, edge is darker
        
        # Core (brightest center)
        glColor4f(1.0, 1.0, 0.98, 1.0)
        gluSphere(self.quadric, self.size * 0.85, 32, 32)
        
        # Mid layer
        glColor4f(1.0, 0.98, 0.92, 0.9)
        gluSphere(self.quadric, self.size * 0.95, 32, 32)
        
        # Edge (limb darkening)
        glColor4f(1.0, 0.95, 0.85, 0.8)
        gluSphere(self.quadric, self.size, 32, 32)
        
        # Corona layers using inverse-square falloff
        # I = I0 / (1 + (r/r0)²)
        corona_layers = [
            (1.3, 0.4, (1.0, 0.98, 0.85)),
            (1.6, 0.25, (1.0, 0.95, 0.75)),
            (2.0, 0.15, (1.0, 0.92, 0.65)),
            (2.5, 0.08, (1.0, 0.88, 0.55)),
            (3.2, 0.04, (1.0, 0.85, 0.50)),
        ]
        
        for scale, alpha, color in corona_layers:
            glPushMatrix()
            glScalef(self.size * scale, self.size * scale, 1)
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(color[0], color[1], color[2], alpha)
            glVertex3f(0, 0, 0)
            glColor4f(color[0], color[1], color[2], 0.0)
            for i in range(49):
                angle = 2 * math.pi * i / 48
                glVertex3f(math.cos(angle), math.sin(angle), 0)
            glEnd()
            glPopMatrix()
        
        # Sun rays (god rays approximation)
        self.draw_rays()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def draw_rays(self):
        """Draw animated sun rays with varying length and intensity"""
        num_rays = 16
        glLineWidth(2.5)
        
        glBegin(GL_LINES)
        for i in range(num_rays):
            # Animated rotation
            base_angle = (i / num_rays) * 2 * math.pi
            angle = base_angle + self.time * 0.1
            
            # Ray length varies with time (breathing effect)
            phase = self.time * 1.2 + i * 0.4
            length_mod = 0.7 + 0.3 * math.sin(phase)
            ray_length = 6.0 * length_mod
            
            # Start just outside sun
            start_dist = self.size * 1.2
            end_dist = start_dist + ray_length
            
            x1 = math.cos(angle) * start_dist
            y1 = math.sin(angle) * start_dist
            x2 = math.cos(angle) * end_dist
            y2 = math.sin(angle) * end_dist
            
            # Intensity varies
            intensity = 0.5 + 0.2 * math.sin(phase * 0.7)
            
            glColor4f(1.0, 0.95, 0.8, intensity * 0.6)
            glVertex3f(x1, y1, 0)
            glColor4f(1.0, 0.9, 0.6, 0.0)
            glVertex3f(x2, y2, 0)
        glEnd()
        glLineWidth(1.0)


class ObamaStatue:
    """
    Highly detailed bronze statue of President Barack Obama
    Uses advanced mesh generation with scipy for realistic anatomy
    Features: Detailed facial features, realistic suit with folds, proper proportions
    """
    def __init__(self, position, scale=1.0):
        self.position = position
        self.scale = scale
        self.quadric = None
        self.display_list = None
        self.marble_texture = None
        
        # Subsurface scattering approximation for bronze
        self.bronze_ambient = [0.25, 0.15, 0.06, 1.0]
        self.bronze_diffuse = [0.75, 0.45, 0.20, 1.0]
        self.bronze_specular = [0.45, 0.35, 0.25, 1.0]
        self.bronze_shininess = 30.0
        
        # Patina (aged bronze - greenish tint in crevices)
        self.patina_ambient = [0.10, 0.18, 0.12, 1.0]
        self.patina_diffuse = [0.20, 0.35, 0.22, 1.0]
        
        # Polished bronze for highlights
        self.polished_ambient = [0.30, 0.20, 0.10, 1.0]
        self.polished_diffuse = [0.85, 0.55, 0.25, 1.0]
        self.polished_specular = [0.65, 0.50, 0.35, 1.0]
        self.polished_shininess = 60.0
        
        # Marble pedestal material - Carrara marble
        self.marble_ambient = [0.22, 0.22, 0.22, 1.0]
        self.marble_diffuse = [0.88, 0.87, 0.85, 1.0]
        self.marble_specular = [0.35, 0.35, 0.35, 1.0]
        self.marble_shininess = 45.0
        
        # Dark marble veins
        self.vein_diffuse = [0.45, 0.45, 0.48, 1.0]
        
        # Gold plaque material - 24k gold
        self.gold_ambient = [0.30, 0.24, 0.08, 1.0]
        self.gold_diffuse = [0.85, 0.70, 0.25, 1.0]
        self.gold_specular = [0.75, 0.65, 0.45, 1.0]
        self.gold_shininess = 75.0
        
        # Pre-generate mesh data
        self._generate_meshes()
        self._create_marble_texture()
        self._compile_display_list()
    
    def _generate_meshes(self):
        """Generate high-quality mesh data for all statue components"""
        # Head mesh vertices using parametric surface
        self.head_mesh = self._generate_head_mesh()
        self.torso_mesh = self._generate_torso_mesh()
        self.suit_details = self._generate_suit_details()
    
    def _generate_head_mesh(self):
        """Generate detailed head mesh with facial features"""
        # Parametric head shape based on superellipsoid
        # f(u,v) = (cos(u)^n * cos(v)^n, sin(u)^n * cos(v)^n, sin(v)^m)
        u = np.linspace(-np.pi, np.pi, 32)
        v = np.linspace(-np.pi/2, np.pi/2, 24)
        U, V = np.meshgrid(u, v)
        
        # Base ellipsoid with Obama's head proportions
        # Slightly elongated, prominent cheekbones
        n1, n2 = 0.9, 0.85
        
        def signed_pow(x, p):
            return np.sign(x) * np.abs(x) ** p
        
        X = 0.16 * signed_pow(np.cos(U), n1) * signed_pow(np.cos(V), n2)
        Y = 0.20 * signed_pow(np.sin(V), n2)  # Taller head
        Z = 0.15 * signed_pow(np.sin(U), n1) * signed_pow(np.cos(V), n2)
        
        # Add facial feature deformations
        # Cheekbones
        cheek_mask = np.exp(-((V - 0.1)**2 / 0.1 + (np.abs(U) - 1.2)**2 / 0.3))
        X += cheek_mask * 0.02
        
        # Jaw definition
        jaw_mask = np.exp(-((V + 0.4)**2 / 0.15)) * (1 - np.abs(U) / np.pi)
        Z += jaw_mask * 0.015
        
        # Brow ridge
        brow_mask = np.exp(-((V - 0.35)**2 / 0.05 + U**2 / 0.8))
        Z += brow_mask * 0.025
        
        return {'X': X, 'Y': Y, 'Z': Z, 'U': U, 'V': V}
    
    def _generate_torso_mesh(self):
        """Generate torso with suit jacket shape"""
        # Suit jacket cross-section varies along height
        heights = np.linspace(0, 1.0, 20)
        angles = np.linspace(0, 2*np.pi, 32)
        
        vertices = []
        normals = []
        
        for i, h in enumerate(heights):
            # Width varies: narrow at waist, wider at shoulders
            t = h  # 0 at bottom, 1 at top
            
            # Shoulder width peaks at t=0.85
            shoulder_factor = np.exp(-((t - 0.85)**2) / 0.05)
            
            # Waist narrowing
            waist_factor = 1.0 - 0.15 * np.exp(-((t - 0.3)**2) / 0.02)
            
            width = 0.22 * (0.8 + 0.4 * t) * waist_factor + shoulder_factor * 0.08
            depth = 0.15 * (0.85 + 0.3 * t)
            
            for angle in angles:
                # Superellipse cross-section for suit shape
                n = 2.5  # Slightly squared
                x = width * np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** (2/n)
                z = depth * np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** (2/n)
                y = h * 0.7  # Scale height
                
                vertices.append([x, y, z])
                
                # Calculate normal
                nx = np.cos(angle)
                nz = np.sin(angle)
                normals.append([nx, 0, nz])
        
        return {'vertices': np.array(vertices), 'normals': np.array(normals)}
    
    def _generate_suit_details(self):
        """Generate suit lapels, buttons, tie, and fabric folds"""
        details = {
            'lapels': [],
            'buttons': [],
            'tie': [],
            'folds': []
        }
        
        # Lapel vertices (V-shape on chest)
        lapel_left = [
            [-0.08, 0.5, 0.13], [-0.15, 0.75, 0.11], [-0.12, 0.75, 0.12],
            [-0.05, 0.5, 0.14], [-0.08, 0.5, 0.13]
        ]
        lapel_right = [
            [0.08, 0.5, 0.13], [0.15, 0.75, 0.11], [0.12, 0.75, 0.12],
            [0.05, 0.5, 0.14], [0.08, 0.5, 0.13]
        ]
        details['lapels'] = [lapel_left, lapel_right]
        
        # Button positions
        details['buttons'] = [
            [0.0, 0.35, 0.14],
            [0.0, 0.25, 0.135]
        ]
        
        # Tie shape (triangular)
        details['tie'] = [
            [0.0, 0.65, 0.135],   # Knot
            [-0.04, 0.55, 0.13],  # Left
            [0.04, 0.55, 0.13],   # Right
            [0.0, 0.2, 0.125],    # Bottom point
        ]
        
        # Fabric fold lines (curved)
        for i in range(5):
            angle = (i - 2) * 0.3
            fold = []
            for t in np.linspace(0, 1, 10):
                x = 0.15 * np.sin(angle) * (1 - t * 0.3)
                y = 0.3 + t * 0.4
                z = 0.1 + 0.02 * np.sin(t * np.pi)
                fold.append([x, y, z])
            details['folds'].append(fold)
        
        return details
    
    def _create_marble_texture(self):
        """Generate procedural marble texture using Perlin noise"""
        size = 256
        
        # Generate base noise
        noise = np.random.rand(size, size)
        
        # Multi-octave noise (turbulence)
        result = np.zeros((size, size))
        amplitude = 1.0
        frequency = 1
        
        for _ in range(6):
            # Smooth the noise
            smoothed = ndimage.gaussian_filter(
                np.random.rand(size // frequency + 1, size // frequency + 1),
                sigma=1
            )
            # Upsample
            upsampled = ndimage.zoom(smoothed, frequency, order=1)[:size, :size]
            result += amplitude * upsampled
            amplitude *= 0.5
            frequency *= 2
        
        # Normalize
        result = (result - result.min()) / (result.max() - result.min())
        
        # Create marble pattern using sine wave distorted by noise
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Veiny marble pattern
        marble = np.sin(X + result * 6)
        marble = (marble + 1) / 2  # Normalize to 0-1
        
        # Add some turbulence
        marble = marble * 0.7 + result * 0.3
        
        # Create RGB image
        pixels = []
        for row in marble:
            for val in row:
                # Base white marble with gray veins
                r = int(255 * (0.85 + 0.15 * val))
                g = int(255 * (0.84 + 0.14 * val))
                b = int(255 * (0.82 + 0.13 * val))
                pixels.extend([r, g, b])
        
        # Create OpenGL texture
        self.marble_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.marble_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, bytes(pixels))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
    
    def _compile_display_list(self):
        """Compile statue geometry into display list for performance"""
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        self._draw_all_geometry()
        glEndList()
    
    def _set_bronze_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.bronze_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.bronze_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.bronze_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.bronze_shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0, 0, 0, 1])
    
    def _set_polished_bronze(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.polished_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.polished_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.polished_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.polished_shininess)
    
    def _set_marble_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.marble_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.marble_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.marble_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.marble_shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0, 0, 0, 1])
    
    def _set_gold_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.gold_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.gold_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.gold_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.gold_shininess)
    
    def _draw_quadric(self):
        if not self.quadric:
            self.quadric = gluNewQuadric()
            gluQuadricNormals(self.quadric, GLU_SMOOTH)
        return self.quadric
    
    def _draw_smooth_cylinder(self, base_r, top_r, height, slices=24, stacks=8):
        """Draw cylinder with smooth shading"""
        q = self._draw_quadric()
        gluCylinder(q, base_r, top_r, height, slices, stacks)
        # Caps
        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(q, 0, base_r, slices, 1)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0, 0, height)
        gluDisk(q, 0, top_r, slices, 1)
        glPopMatrix()
    
    def _draw_sphere(self, radius, slices=24, stacks=24):
        q = self._draw_quadric()
        gluSphere(q, radius, slices, stacks)
    
    def _draw_superellipsoid(self, rx, ry, rz, n1, n2, slices=24, stacks=16):
        """Draw superellipsoid for more natural shapes"""
        def signed_pow(x, p):
            return np.sign(x) * np.abs(x) ** p
        
        u = np.linspace(-np.pi, np.pi, slices)
        v = np.linspace(-np.pi/2, np.pi/2, stacks)
        
        for i in range(len(v) - 1):
            glBegin(GL_QUAD_STRIP)
            for j in range(len(u)):
                for vi in [v[i], v[i+1]]:
                    cu, su = np.cos(u[j]), np.sin(u[j])
                    cv, sv = np.cos(vi), np.sin(vi)
                    
                    x = rx * signed_pow(cv, n1) * signed_pow(cu, n2)
                    y = ry * signed_pow(sv, n1)
                    z = rz * signed_pow(cv, n1) * signed_pow(su, n2)
                    
                    # Normal (approximate)
                    nx = signed_pow(cv, 2-n1) * signed_pow(cu, 2-n2) / rx
                    ny = signed_pow(sv, 2-n1) / ry
                    nz = signed_pow(cv, 2-n1) * signed_pow(su, 2-n2) / rz
                    
                    mag = np.sqrt(nx*nx + ny*ny + nz*nz) + 0.001
                    glNormal3f(nx/mag, ny/mag, nz/mag)
                    glVertex3f(x, y, z)
            glEnd()
    
    def _draw_box(self, width, height, depth):
        """Draw a rectangular box with proper normals"""
        w, h, d = width/2, height/2, depth/2
        
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, 0, 1)
        glTexCoord2f(0, 0); glVertex3f(-w, -h, d)
        glTexCoord2f(1, 0); glVertex3f(w, -h, d)
        glTexCoord2f(1, 1); glVertex3f(w, h, d)
        glTexCoord2f(0, 1); glVertex3f(-w, h, d)
        # Back
        glNormal3f(0, 0, -1)
        glTexCoord2f(0, 0); glVertex3f(w, -h, -d)
        glTexCoord2f(1, 0); glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 1); glVertex3f(-w, h, -d)
        glTexCoord2f(0, 1); glVertex3f(w, h, -d)
        # Top
        glNormal3f(0, 1, 0)
        glTexCoord2f(0, 0); glVertex3f(-w, h, d)
        glTexCoord2f(1, 0); glVertex3f(w, h, d)
        glTexCoord2f(1, 1); glVertex3f(w, h, -d)
        glTexCoord2f(0, 1); glVertex3f(-w, h, -d)
        # Bottom
        glNormal3f(0, -1, 0)
        glTexCoord2f(0, 0); glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 0); glVertex3f(w, -h, -d)
        glTexCoord2f(1, 1); glVertex3f(w, -h, d)
        glTexCoord2f(0, 1); glVertex3f(-w, -h, d)
        # Right
        glNormal3f(1, 0, 0)
        glTexCoord2f(0, 0); glVertex3f(w, -h, d)
        glTexCoord2f(1, 0); glVertex3f(w, -h, -d)
        glTexCoord2f(1, 1); glVertex3f(w, h, -d)
        glTexCoord2f(0, 1); glVertex3f(w, h, d)
        # Left
        glNormal3f(-1, 0, 0)
        glTexCoord2f(0, 0); glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 0); glVertex3f(-w, -h, d)
        glTexCoord2f(1, 1); glVertex3f(-w, h, d)
        glTexCoord2f(0, 1); glVertex3f(-w, h, -d)
        glEnd()
    
    def _draw_pedestal(self):
        """Draw detailed marble pedestal with texture"""
        self._set_marble_material()
        glColor4f(0.92, 0.91, 0.88, 1.0)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.marble_texture)
        
        # Base tier with chamfered edges
        glPushMatrix()
        glTranslatef(0, 0.3, 0)
        self._draw_box(2.2, 0.6, 2.2)
        glPopMatrix()
        
        # Decorative molding
        glPushMatrix()
        glTranslatef(0, 0.65, 0)
        self._draw_box(2.0, 0.1, 2.0)
        glPopMatrix()
        
        # Middle tier
        glPushMatrix()
        glTranslatef(0, 0.9, 0)
        self._draw_box(1.8, 0.4, 1.8)
        glPopMatrix()
        
        # Upper molding
        glPushMatrix()
        glTranslatef(0, 1.15, 0)
        self._draw_box(1.6, 0.1, 1.6)
        glPopMatrix()
        
        # Top tier
        glPushMatrix()
        glTranslatef(0, 1.35, 0)
        self._draw_box(1.5, 0.3, 1.5)
        glPopMatrix()
        
        glDisable(GL_TEXTURE_2D)
        
        # Gold plaque
        self._set_gold_material()
        glColor4f(0.85, 0.68, 0.20, 1.0)
        glPushMatrix()
        glTranslatef(0, 0.65, 1.11)
        self._draw_box(1.1, 0.35, 0.03)
        glPopMatrix()
        
        # Plaque border
        glPushMatrix()
        glTranslatef(0, 0.65, 1.125)
        glColor4f(0.75, 0.58, 0.15, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-0.52, -0.15, 0)
        glVertex3f(0.52, -0.15, 0)
        glVertex3f(0.52, 0.15, 0)
        glVertex3f(-0.52, 0.15, 0)
        glEnd()
        glLineWidth(1.0)
        glPopMatrix()
    
    def _draw_detailed_figure(self):
        """Draw highly detailed Obama figure"""
        self._set_bronze_material()
        glColor4f(0.60, 0.38, 0.18, 1.0)
        
        base_y = 1.5
        
        # === FEET / SHOES ===
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.14, base_y + 0.05, 0.06)
            glScalef(0.9, 0.35, 1.5)
            self._draw_superellipsoid(0.1, 0.1, 0.1, 0.5, 0.5, 16, 12)
            glPopMatrix()
        
        # === LEGS (with pants creases) ===
        for side in [-1, 1]:
            # Upper leg
            glPushMatrix()
            glTranslatef(side * 0.14, base_y + 0.08, 0)
            glRotatef(-90, 1, 0, 0)
            self._draw_smooth_cylinder(0.095, 0.085, 0.45, 20, 6)
            glPopMatrix()
            
            # Lower leg
            glPushMatrix()
            glTranslatef(side * 0.14, base_y + 0.5, 0)
            glRotatef(-90, 1, 0, 0)
            self._draw_smooth_cylinder(0.085, 0.075, 0.4, 20, 6)
            glPopMatrix()
        
        # === TORSO (suit jacket with shape) ===
        # Lower torso / hips
        glPushMatrix()
        glTranslatef(0, base_y + 0.95, 0)
        self._draw_superellipsoid(0.20, 0.12, 0.13, 0.7, 0.7, 24, 16)
        glPopMatrix()
        
        # Mid torso
        glPushMatrix()
        glTranslatef(0, base_y + 1.1, 0)
        self._draw_superellipsoid(0.22, 0.15, 0.14, 0.6, 0.6, 24, 16)
        glPopMatrix()
        
        # Upper torso / chest
        glPushMatrix()
        glTranslatef(0, base_y + 1.28, 0)
        self._draw_superellipsoid(0.25, 0.15, 0.15, 0.55, 0.55, 24, 16)
        glPopMatrix()
        
        # Shoulders
        glPushMatrix()
        glTranslatef(0, base_y + 1.42, 0)
        self._draw_superellipsoid(0.32, 0.08, 0.12, 0.4, 0.6, 24, 12)
        glPopMatrix()
        
        # === SUIT DETAILS ===
        # Lapels
        self._set_polished_bronze()
        for side in [-1, 1]:
            glBegin(GL_TRIANGLE_STRIP)
            glNormal3f(0, 0, 1)
            glVertex3f(side * 0.03, base_y + 1.0, 0.145)
            glVertex3f(side * 0.08, base_y + 1.0, 0.14)
            glVertex3f(side * 0.02, base_y + 1.35, 0.155)
            glVertex3f(side * 0.15, base_y + 1.4, 0.12)
            glEnd()
        
        # Tie
        self._set_bronze_material()
        glBegin(GL_TRIANGLES)
        glNormal3f(0, 0, 1)
        # Knot
        glVertex3f(-0.025, base_y + 1.38, 0.155)
        glVertex3f(0.025, base_y + 1.38, 0.155)
        glVertex3f(0.0, base_y + 1.35, 0.16)
        # Tie body
        glVertex3f(-0.035, base_y + 1.35, 0.155)
        glVertex3f(0.035, base_y + 1.35, 0.155)
        glVertex3f(0.0, base_y + 0.95, 0.145)
        glEnd()
        
        # Buttons (small spheres)
        self._set_polished_bronze()
        for y_off in [0.15, 0.05]:
            glPushMatrix()
            glTranslatef(0, base_y + 1.0 + y_off, 0.145)
            self._draw_sphere(0.018, 12, 12)
            glPopMatrix()
        
        # === ARMS ===
        self._set_bronze_material()
        
        # Left arm (down at side)
        # Upper arm
        glPushMatrix()
        glTranslatef(-0.32, base_y + 1.35, 0)
        glRotatef(8, 0, 0, 1)
        glRotatef(-90, 1, 0, 0)
        self._draw_smooth_cylinder(0.065, 0.055, 0.32, 16, 4)
        glPopMatrix()
        
        # Left forearm
        glPushMatrix()
        glTranslatef(-0.36, base_y + 1.05, 0.02)
        glRotatef(5, 0, 0, 1)
        glRotatef(-85, 1, 0, 0)
        self._draw_smooth_cylinder(0.055, 0.045, 0.28, 16, 4)
        glPopMatrix()
        
        # Left hand
        glPushMatrix()
        glTranslatef(-0.38, base_y + 0.78, 0.02)
        glScalef(0.8, 1.2, 0.5)
        self._draw_sphere(0.055, 12, 12)
        glPopMatrix()
        
        # Right arm (raised, gesturing)
        # Upper arm
        glPushMatrix()
        glTranslatef(0.32, base_y + 1.35, 0.02)
        glRotatef(-15, 0, 0, 1)
        glRotatef(-60, 1, 0, 0)
        self._draw_smooth_cylinder(0.065, 0.055, 0.30, 16, 4)
        glPopMatrix()
        
        # Right forearm
        glPushMatrix()
        glTranslatef(0.38, base_y + 1.12, 0.22)
        glRotatef(-30, 1, 0, 0)
        glRotatef(10, 0, 0, 1)
        glRotatef(-90, 1, 0, 0)
        self._draw_smooth_cylinder(0.055, 0.045, 0.25, 16, 4)
        glPopMatrix()
        
        # Right hand (open, gesturing - more detailed)
        glPushMatrix()
        glTranslatef(0.40, base_y + 1.05, 0.42)
        glRotatef(-20, 1, 0, 0)
        # Palm
        glScalef(1.0, 0.5, 1.1)
        self._draw_sphere(0.055, 12, 12)
        glPopMatrix()
        
        # Fingers (simplified)
        for i, offset in enumerate([-0.025, -0.01, 0.005, 0.02]):
            glPushMatrix()
            glTranslatef(0.40 + offset, base_y + 1.03, 0.47 + i * 0.01)
            glRotatef(-30, 1, 0, 0)
            glRotatef(-90, 1, 0, 0)
            self._draw_smooth_cylinder(0.012, 0.008, 0.06, 8, 2)
            glPopMatrix()
        
        # === NECK ===
        glPushMatrix()
        glTranslatef(0, base_y + 1.48, 0)
        glRotatef(-90, 1, 0, 0)
        self._draw_smooth_cylinder(0.07, 0.065, 0.12, 16, 3)
        glPopMatrix()
        
        # === HEAD (detailed) ===
        self._draw_detailed_head(base_y + 1.72)
    
    def _draw_detailed_head(self, base_y):
        """Draw detailed head with facial features"""
        # Main cranium using mesh data
        mesh = self.head_mesh
        
        glPushMatrix()
        glTranslatef(0, base_y, 0)
        
        # Draw head mesh as quad strips
        X, Y, Z = mesh['X'], mesh['Y'], mesh['Z']
        rows, cols = X.shape
        
        for i in range(rows - 1):
            glBegin(GL_QUAD_STRIP)
            for j in range(cols):
                for di in [0, 1]:
                    x, y, z = X[i+di, j], Y[i+di, j], Z[i+di, j]
                    # Calculate normal (approximate)
                    nx, ny, nz = x, y, z
                    mag = np.sqrt(nx*nx + ny*ny + nz*nz) + 0.001
                    glNormal3f(nx/mag, ny/mag, nz/mag)
                    glVertex3f(x, y, z)
            glEnd()
        
        glPopMatrix()
        
        # Ears (more detailed)
        self._set_polished_bronze()
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.155, base_y - 0.02, 0)
            glScalef(0.25, 0.5, 0.4)
            self._draw_sphere(0.1, 16, 12)
            glPopMatrix()
            
            # Ear detail (inner curve)
            glPushMatrix()
            glTranslatef(side * 0.15, base_y - 0.02, 0.01)
            glScalef(0.15, 0.35, 0.25)
            self._set_bronze_material()
            self._draw_sphere(0.08, 12, 8)
            glPopMatrix()
        
        # Facial features
        self._set_bronze_material()
        
        # Nose
        glPushMatrix()
        glTranslatef(0, base_y - 0.03, 0.14)
        glScalef(0.5, 0.7, 0.8)
        self._draw_superellipsoid(0.04, 0.05, 0.04, 0.6, 0.6, 12, 10)
        glPopMatrix()
        
        # Nose bridge
        glPushMatrix()
        glTranslatef(0, base_y + 0.04, 0.13)
        glScalef(0.35, 1.2, 0.5)
        self._draw_superellipsoid(0.03, 0.04, 0.03, 0.5, 0.5, 10, 8)
        glPopMatrix()
        
        # Nostrils (subtle indentations represented as small spheres)
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.02, base_y - 0.06, 0.135)
            self._draw_sphere(0.015, 8, 8)
            glPopMatrix()
        
        # Brow ridge
        glPushMatrix()
        glTranslatef(0, base_y + 0.1, 0.11)
        glScalef(2.2, 0.35, 0.6)
        self._draw_superellipsoid(0.05, 0.04, 0.04, 0.4, 0.4, 16, 8)
        glPopMatrix()
        
        # Eyes (recessed area with brow shadow)
        for side in [-1, 1]:
            # Eye socket shadow
            glPushMatrix()
            glTranslatef(side * 0.055, base_y + 0.05, 0.115)
            glScalef(1.3, 0.6, 0.6)
            self._draw_sphere(0.028, 12, 10)
            glPopMatrix()
        
        # Cheekbones (prominent)
        self._set_polished_bronze()
        for side in [-1, 1]:
            glPushMatrix()
            glTranslatef(side * 0.11, base_y - 0.01, 0.08)
            glScalef(0.8, 0.5, 0.7)
            self._draw_sphere(0.045, 14, 12)
            glPopMatrix()
        
        # Jaw line
        self._set_bronze_material()
        glPushMatrix()
        glTranslatef(0, base_y - 0.13, 0.06)
        glScalef(1.0, 0.5, 0.8)
        self._draw_superellipsoid(0.12, 0.06, 0.08, 0.5, 0.5, 16, 10)
        glPopMatrix()
        
        # Chin
        glPushMatrix()
        glTranslatef(0, base_y - 0.16, 0.09)
        glScalef(0.7, 0.5, 0.7)
        self._draw_sphere(0.05, 14, 12)
        glPopMatrix()
        
        # Lips area
        glPushMatrix()
        glTranslatef(0, base_y - 0.08, 0.13)
        glScalef(1.3, 0.5, 0.6)
        self._draw_superellipsoid(0.035, 0.025, 0.025, 0.5, 0.5, 12, 8)
        glPopMatrix()
        
        # Hair (short, textured) - represented as slight bump on top
        self._set_bronze_material()
        glPushMatrix()
        glTranslatef(0, base_y + 0.14, -0.01)
        glScalef(0.95, 0.4, 0.9)
        self._draw_superellipsoid(0.14, 0.1, 0.12, 0.6, 0.6, 20, 14)
        glPopMatrix()
        
        # Hair texture lines
        glColor4f(0.45, 0.28, 0.12, 1.0)
        glLineWidth(1.0)
        for i in range(12):
            angle = (i / 12) * np.pi - np.pi/2
            glBegin(GL_LINE_STRIP)
            for t in np.linspace(0, 1, 8):
                r = 0.14 * (1 - t * 0.1)
                x = r * np.sin(angle + t * 0.2)
                y = base_y + 0.14 + t * 0.03
                z = -0.01 + r * np.cos(angle + t * 0.1) * 0.7
                glVertex3f(x, y, z)
            glEnd()
        glColor4f(0.60, 0.38, 0.18, 1.0)
    
    def _draw_all_geometry(self):
        """Draw all statue components"""
        self._draw_pedestal()
        self._draw_detailed_figure()
    
    def draw(self):
        """Draw the complete statue using display list"""
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glScalef(self.scale, self.scale, self.scale)
        
        # Use display list for performance
        if self.display_list:
            glCallList(self.display_list)
        else:
            self._draw_all_geometry()
        
        glPopMatrix()
        
        # Reset
        glColor4f(1, 1, 1, 1)
        glMaterialfv(GL_FRONT, GL_EMISSION, [0, 0, 0, 1])


class ShadowSystem:
    """Shadow system implementation with physically-based rendering"""
    def __init__(self, light_pos):
        self.light_pos = light_pos
        self.light_radius = 2.5
        # Pre-compute light direction unit vector
        mag = math.sqrt(light_pos[0]**2 + light_pos[1]**2 + light_pos[2]**2)
        self.light_dir = (light_pos[0]/mag, light_pos[1]/mag, light_pos[2]/mag)
        # Shadow display list for performance
        self.shadow_circle_list = None
        self._create_shadow_geometry()
    
    def _create_shadow_geometry(self):
        """Pre-compile shadow circle geometry"""
        self.shadow_circle_list = glGenLists(1)
        glNewList(self.shadow_circle_list, GL_COMPILE)
        segments = 32
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)  # Center
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(math.cos(angle), 0, math.sin(angle))
        glEnd()
        glEndList()
    
    def _calculate_penumbra(self, ball_radius, ball_height):
        """
        Calculate penumbra ratio using the penumbra cone formula:
        penumbra_size = (light_radius / light_distance) * shadow_distance
        
        For a spherical occluder:
        - Umbra shrinks as shadow moves away from object
        - Penumbra grows as shadow moves away from object
        """
        light_distance = math.sqrt(sum(x**2 for x in self.light_pos))
        angular_size = self.light_radius / light_distance
        
        # Shadow cone angle (penumbra outer edge)
        penumbra_angle = math.atan2(self.light_radius + ball_radius, light_distance)
        # Umbra cone angle (sharp inner edge)
        umbra_angle = math.atan2(self.light_radius - ball_radius, light_distance)
        
        # Penumbra ratio increases with height (longer shadow travel distance)
        penumbra_ratio = 1.0 + ball_height * angular_size * 0.5
        
        return penumbra_ratio, max(0.1, 1.0 - ball_height * 0.02)
    
    def _calculate_shadow_projection(self, ball_pos, ball_radius):
        """
        Project shadow onto ground plane using homogeneous coordinates
        Shadow projection matrix derivation from light-plane intersection
        
        For a point P and light L projecting onto plane y=0:
        shadow_x = P.x - L.x * (P.y / L.y)
        shadow_z = P.z - L.z * (P.y / L.y)
        
        With elliptical distortion based on light angle
        """
        lx, ly, lz = self.light_pos
        bx, by, bz = ball_pos.x, ball_pos.y, ball_pos.z
        
        # Projection ratio (intersection with y=0 plane)
        if ly <= 0:
            return bx, bz, ball_radius, ball_radius
        
        t = by / ly
        
        # Shadow center position
        shadow_x = bx - lx * t
        shadow_z = bz - lz * t
        
        # Elliptical shadow based on sun angle
        # Major axis aligned with light direction projected onto ground
        light_angle_xz = math.atan2(lz, lx)
        light_elevation = math.atan2(ly, math.sqrt(lx*lx + lz*lz))
        
        # Shadow stretches along light direction based on elevation angle
        # cos(elevation) gives stretch factor (perpendicular = circle, grazing = long ellipse)
        stretch_factor = 1.0 / max(0.3, math.sin(light_elevation))
        
        # Shadow grows with height due to penumbra
        base_size = ball_radius * (1.0 + by * 0.08)
        
        # Ellipse semi-axes
        semi_major = base_size * stretch_factor
        semi_minor = base_size
        
        return shadow_x, shadow_z, semi_major, semi_minor, light_angle_xz
    
    def _exponential_falloff(self, distance, decay_rate):
        """
        Physically-based exponential light falloff
        I = I0 * e^(-k * d)
        """
        return math.exp(-decay_rate * distance)
    
    def _calculate_contact_shadow_intensity(self, ball_height, ball_radius):
        """
        Ambient occlusion approximation for contact shadow
        Using hemisphere visibility estimation
        
        AO ≈ 1 - (h / (h + r))^2 where h = height above ground, r = radius
        """
        h = max(0, ball_height - ball_radius)
        r = ball_radius
        if h < r * 2:
            visibility = (h / (h + r)) ** 2
            return 1.0 - visibility
        return 0.0
    
    def draw_ball_shadow(self, ball):
        """Draw physically accurate shadow with multiple layers"""
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)  # Don't write to depth buffer
        
        bx, by, bz = ball.position.x, ball.position.y, ball.position.z
        
        # Calculate shadow projection
        proj = self._calculate_shadow_projection(ball.position, ball.radius)
        shadow_x, shadow_z, semi_major, semi_minor, rotation = proj
        
        # Calculate penumbra and intensity
        penumbra_ratio, base_intensity = self._calculate_penumbra(ball.radius, by)
        contact_ao = self._calculate_contact_shadow_intensity(by, ball.radius)
        
        # Height-based fade using inverse square approximation
        height_fade = 1.0 / (1.0 + by * 0.15)
        
        # Final shadow intensity with atmospheric scattering consideration
        # Shadows become less dark due to scattered light filling them
        atmospheric_fill = 0.15 * (1.0 - self._exponential_falloff(by, 0.1))
        shadow_intensity = max(0.05, base_intensity * height_fade * 0.6 - atmospheric_fill)
        
        glPushMatrix()
        glTranslatef(shadow_x, 0.002, shadow_z)
        glRotatef(math.degrees(rotation), 0, 1, 0)
        
        # Layer 1: Soft outer penumbra (largest, most transparent)
        penumbra_size = penumbra_ratio * 1.4
        glPushMatrix()
        glScalef(semi_major * penumbra_size, 1, semi_minor * penumbra_size)
        segments = 32
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.0, 0.02, 0.0, shadow_intensity * 0.15)
        glVertex3f(0, 0, 0)
        glColor4f(0.0, 0.02, 0.0, 0.0)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(math.cos(angle), 0, math.sin(angle))
        glEnd()
        glPopMatrix()
        
        # Layer 2: Mid penumbra
        mid_size = penumbra_ratio * 1.15
        glPushMatrix()
        glScalef(semi_major * mid_size, 1, semi_minor * mid_size)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.0, 0.015, 0.0, shadow_intensity * 0.3)
        glVertex3f(0, 0, 0)
        glColor4f(0.0, 0.015, 0.0, 0.0)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(math.cos(angle), 0, math.sin(angle))
        glEnd()
        glPopMatrix()
        
        # Layer 3: Core shadow (umbra)
        glPushMatrix()
        glScalef(semi_major, 1, semi_minor)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.0, 0.01, 0.0, shadow_intensity * 0.5)
        glVertex3f(0, 0, 0)
        glColor4f(0.0, 0.01, 0.0, shadow_intensity * 0.1)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(math.cos(angle), 0, math.sin(angle))
        glEnd()
        glPopMatrix()
        
        glPopMatrix()
        
        # Layer 4: Contact shadow (ambient occlusion at ground contact)
        if contact_ao > 0.01:
            contact_size = ball.radius * 0.8 * (1.0 + (1.0 - contact_ao) * 0.5)
            glPushMatrix()
            glTranslatef(bx, 0.003, bz)
            glScalef(contact_size, 1, contact_size)
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.0, 0.005, 0.0, contact_ao * 0.7)
            glVertex3f(0, 0, 0)
            glColor4f(0.0, 0.005, 0.0, 0.0)
            for i in range(24 + 1):
                angle = 2 * math.pi * i / 24
                glVertex3f(math.cos(angle), 0, math.sin(angle))
            glEnd()
            glPopMatrix()
        
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)


class GameEngine:
    """Main 3D Game Engine - Hyperrealistic"""
    def __init__(self):
        # Enable VSync for smooth rendering, no FPS cap
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)  # VSync
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL | pygame.HWSURFACE)
        pygame.display.set_caption("3D Engine - Hyperrealistic Silver Ball")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.init_opengl()
        
        # Wind system
        self.wind = Wind()
        
        # Game objects
        self.camera = Camera()
        self.ball = ReflectiveBall(Vector3(0, 5, 0), radius=0.5)
        self.ground = GrassGround(size=80, wind=self.wind)
        self.sun = Sun()
        self.shadow_system = ShadowSystem(SUN_POSITION)
        
        # Obama statue
        self.obama_statue = ObamaStatue(Vector3(-8, 0, -10), scale=1.8)
        
        # Create environment map
        self.env_texture = self.create_environment_map()
        
        # Timing - no FPS cap
        self.clock = pygame.time.Clock()
        self.running = True
        self.last_time = pygame.time.get_ticks() / 1000.0
        self.fps_display = 0
        self.fps_timer = 0
    
    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)  # Fill light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        
        # Better quality hints
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(FOV, SCREEN_WIDTH/SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        
        # Main sun light - warm, realistic
        glLightfv(GL_LIGHT0, GL_POSITION, [SUN_POSITION[0], SUN_POSITION[1], SUN_POSITION[2], 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.15, 0.15, 0.12, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.95, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 0.98, 0.9, 1.0])
        
        # Sky fill light (blue bounce light from sky)
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 50.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.08, 0.12, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.2, 0.25, 0.35, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.1, 0.12, 0.15, 1.0])
        
        # Global ambient - subtle
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.12, 0.14, 0.12, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        
        # Atmospheric fog - realistic haze
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, [0.7, 0.82, 0.95, 1.0])
        glFogf(GL_FOG_DENSITY, 0.004)
        glHint(GL_FOG_HINT, GL_NICEST)
    
    def create_environment_map(self):
        """
        Create high-quality HDR-style environment map for physically accurate reflections
        Uses spherical coordinate mapping with:
        - Rayleigh scattering for sky color
        - Physically-based sun with limb darkening
        - Fresnel-weighted grass reflections
        """
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        size = 512
        pixels = []
        
        # Sun parameters
        sun_theta = 0.75 * math.pi  # Azimuth (horizontal angle)
        sun_phi = 0.25 * math.pi    # Elevation (vertical angle from horizon)
        sun_radius = 0.08           # Angular radius
        
        for y in range(size):
            for x in range(size):
                # Convert to spherical coordinates for environment mapping
                # u, v in [0, 1] -> theta, phi for sphere
                u = x / size
                v = y / size
                
                # Map to sphere (equirectangular projection)
                theta = u * 2 * math.pi  # 0 to 2π
                phi = v * math.pi        # 0 to π (top to bottom)
                
                # Direction vector on unit sphere
                dx = math.sin(phi) * math.cos(theta)
                dy = math.cos(phi)  # Up is +Y
                dz = math.sin(phi) * math.sin(theta)
                
                # Sky color using Rayleigh scattering approximation
                # I(θ) ∝ (1 + cos²θ) for Rayleigh phase function
                # Blue scatters more: I ∝ λ^(-4)
                
                if dy > -0.1:  # Above horizon
                    # Height in sky (0 at horizon, 1 at zenith)
                    sky_height = max(0, dy)
                    
                    # Rayleigh scattering - blue dominates at zenith
                    # At horizon, path length is longer, more scattering, whiter/orange
                    wavelength_red = 0.65
                    wavelength_green = 0.55
                    wavelength_blue = 0.45
                    
                    # Scattering coefficient ∝ 1/λ^4
                    scatter_r = 1.0 / (wavelength_red ** 4)
                    scatter_g = 1.0 / (wavelength_green ** 4)
                    scatter_b = 1.0 / (wavelength_blue ** 4)
                    
                    # Normalize
                    max_scatter = max(scatter_r, scatter_g, scatter_b)
                    scatter_r /= max_scatter
                    scatter_g /= max_scatter
                    scatter_b /= max_scatter
                    
                    # Optical depth increases at horizon (longer path through atmosphere)
                    # Using Chapman function approximation
                    if sky_height > 0.01:
                        optical_depth = 1.0 / (sky_height + 0.1)
                    else:
                        optical_depth = 8.0
                    optical_depth = min(optical_depth, 10.0)
                    
                    # Sky color calculation
                    # More blue at zenith, more white/yellow at horizon
                    zenith_factor = sky_height ** 0.5
                    horizon_factor = 1.0 - zenith_factor
                    
                    r = 0.15 + 0.20 * horizon_factor + 0.15 * zenith_factor
                    g = 0.35 + 0.35 * horizon_factor + 0.25 * zenith_factor
                    b = 0.65 + 0.30 * horizon_factor + 0.30 * zenith_factor * scatter_b
                    
                    # Apply scattering
                    r = r * (0.7 + 0.3 * scatter_r)
                    g = g * (0.8 + 0.2 * scatter_g)
                    b = b * (0.6 + 0.4 * scatter_b)
                    
                    # Sun calculation with limb darkening
                    # Angular distance from sun
                    sun_dx = math.sin(sun_phi) * math.cos(sun_theta)
                    sun_dy = math.cos(sun_phi)
                    sun_dz = math.sin(sun_phi) * math.sin(sun_theta)
                    
                    # Dot product for angular distance
                    cos_angle = dx * sun_dx + dy * sun_dy + dz * sun_dz
                    angle_to_sun = math.acos(max(-1, min(1, cos_angle)))
                    
                    if angle_to_sun < sun_radius * 3:
                        # Inside sun disk or corona
                        if angle_to_sun < sun_radius:
                            # Sun disk with limb darkening
                            # I(r) = I0 * (1 - u * (1 - sqrt(1 - r²)))
                            # where u ≈ 0.6 for solar limb darkening
                            r_norm = angle_to_sun / sun_radius
                            limb_darkening = 1.0 - 0.6 * (1.0 - math.sqrt(max(0, 1.0 - r_norm * r_norm)))
                            
                            # Sun color (slightly warm white)
                            r = min(1.0, 1.0 * limb_darkening + r * 0.1)
                            g = min(1.0, 0.98 * limb_darkening + g * 0.1)
                            b = min(1.0, 0.9 * limb_darkening + b * 0.1)
                        else:
                            # Corona/glow using Gaussian falloff
                            # I = I0 * exp(-(r/σ)²)
                            sigma = sun_radius * 1.5
                            glow = math.exp(-((angle_to_sun - sun_radius) / sigma) ** 2)
                            
                            r = min(1.0, r + glow * 0.8)
                            g = min(1.0, g + glow * 0.7)
                            b = min(1.0, b + glow * 0.4)
                    
                    # Horizon glow (atmospheric scattering near horizon)
                    if sky_height < 0.2:
                        horizon_glow = (0.2 - sky_height) / 0.2
                        horizon_glow = horizon_glow ** 2
                        r = min(1.0, r + horizon_glow * 0.2)
                        g = min(1.0, g + horizon_glow * 0.15)
                        b = min(1.0, b + horizon_glow * 0.1)
                
                else:
                    # Below horizon - grass/ground reflection
                    ground_depth = -dy  # How far below horizon (0 to 1)
                    
                    # Grass color with variation
                    # Using Perlin-like noise approximation
                    noise1 = math.sin(theta * 8 + phi * 5) * 0.5 + 0.5
                    noise2 = math.sin(theta * 23 + phi * 17) * 0.3 + 0.5
                    noise3 = math.cos(theta * 11 - phi * 7) * 0.2 + 0.5
                    noise = (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2)
                    
                    # Grass gets darker further from horizon (viewing more grass, less sky reflection)
                    base_r = 0.12 + noise * 0.08
                    base_g = 0.32 + noise * 0.18
                    base_b = 0.06 + noise * 0.04
                    
                    # Darken with depth
                    depth_factor = 1.0 - ground_depth * 0.4
                    r = base_r * depth_factor
                    g = base_g * depth_factor
                    b = base_b * depth_factor
                    
                    # Slight blue tint from sky reflection on grass (Fresnel)
                    fresnel = 0.04 + 0.96 * ((1 - ground_depth) ** 5)
                    r = r * (1 - fresnel * 0.1) + 0.5 * fresnel * 0.1
                    g = g * (1 - fresnel * 0.1) + 0.7 * fresnel * 0.1
                    b = b * (1 - fresnel * 0.1) + 0.9 * fresnel * 0.1
                
                # Convert to 8-bit
                pixels.extend([
                    int(max(0, min(255, r * 255))),
                    int(max(0, min(255, g * 255))),
                    int(max(0, min(255, b * 255)))
                ])
        
        pixel_data = bytes(pixels)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        return texture
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_r:
                    self.ball.position = Vector3(0, 5, 0)
                    self.ball.velocity = Vector3()
                elif event.key == K_f:
                    self.ball.apply_force(Vector3(
                        self.camera.front.x * 500,
                        300,
                        self.camera.front.z * 500
                    ))
            elif event.type == MOUSEMOTION:
                xoffset, yoffset = event.rel
                self.camera.process_mouse(xoffset, -yoffset)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    direction = (self.ball.position - self.camera.position).normalize()
                    self.ball.apply_force(direction * 400)
    
    def update(self, delta_time):
        # Clamp delta_time to prevent physics explosions on lag spikes
        delta_time = min(delta_time, 0.1)
        
        keys = pygame.key.get_pressed()
        self.camera.update(delta_time, keys)
        
        # Update wind
        self.wind.update(delta_time)
        
        # Apply light wind force to ball
        wind_force = self.wind.get_wind_force()
        self.ball.apply_force(wind_force)
        
        self.ball.update(delta_time)
        self.sun.update(delta_time)
        
        if self.ball.position.y < -10:
            self.ball.position = Vector3(0, 5, 0)
            self.ball.velocity = Vector3()
    
    def draw_skybox(self):
        """Draw hyperrealistic gradient sky"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Multi-band sky gradient for realism
        # Top of sky - deep blue
        glBegin(GL_QUADS)
        glColor3f(0.2, 0.45, 0.9)  # Deep sky blue
        glVertex3f(-1, 1, -0.999)
        glVertex3f(1, 1, -0.999)
        glColor3f(0.35, 0.6, 0.95)  # Mid sky
        glVertex3f(1, 0.4, -0.999)
        glVertex3f(-1, 0.4, -0.999)
        glEnd()
        
        # Middle sky
        glBegin(GL_QUADS)
        glColor3f(0.35, 0.6, 0.95)
        glVertex3f(-1, 0.4, -0.999)
        glVertex3f(1, 0.4, -0.999)
        glColor3f(0.65, 0.82, 1.0)  # Horizon glow
        glVertex3f(1, 0.0, -0.999)
        glVertex3f(-1, 0.0, -0.999)
        glEnd()
        
        # Horizon band - warm atmospheric glow
        glBegin(GL_QUADS)
        glColor3f(0.65, 0.82, 1.0)
        glVertex3f(-1, 0.0, -0.999)
        glVertex3f(1, 0.0, -0.999)
        glColor3f(0.85, 0.9, 0.95)  # Bright horizon
        glVertex3f(1, -0.1, -0.999)
        glVertex3f(-1, -0.1, -0.999)
        glEnd()
        
        # Below horizon - fades to ground
        glBegin(GL_QUADS)
        glColor3f(0.75, 0.85, 0.85)
        glVertex3f(-1, -0.1, -0.999)
        glVertex3f(1, -0.1, -0.999)
        glColor3f(0.4, 0.5, 0.35)  # Distant grass hint
        glVertex3f(1, -1, -0.999)
        glVertex3f(-1, -1, -0.999)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_FOG)
    
    def draw_sun_rays_screen(self):
        """Draw volumetric sun rays effect - optimized"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        sun_screen_x = SCREEN_WIDTH * 0.8
        sun_screen_y = SCREEN_HEIGHT * 0.12
        
        num_rays = 16  # Reduced from 24
        glBegin(GL_TRIANGLES)
        for i in range(num_rays):
            angle = (i / num_rays) * math.pi * 0.7 + math.pi * 0.65
            length = 450 + math.sin(self.sun.time * 1.8 + i * 0.4) * 60
            glColor4f(1.0, 0.95, 0.75, 0.02)
            glVertex2f(sun_screen_x, sun_screen_y)
            glColor4f(1.0, 0.9, 0.6, 0.0)
            glVertex2f(sun_screen_x + math.cos(angle - 0.04) * length,
                      sun_screen_y + math.sin(angle - 0.04) * length)
            glVertex2f(sun_screen_x + math.cos(angle + 0.04) * length,
                      sun_screen_y + math.sin(angle + 0.04) * length)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def draw_hud(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        size = 10
        glColor3f(1, 1, 1)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()
        glLineWidth(1)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.65, 0.78, 0.92, 1.0)
        
        self.draw_skybox()
        
        glLoadIdentity()
        view_params = self.camera.get_view_matrix_params()
        gluLookAt(*view_params)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [SUN_POSITION[0], SUN_POSITION[1], SUN_POSITION[2], 1.0])
        
        # Draw sun
        self.sun.draw()
        
        # Draw shadows
        self.shadow_system.draw_ball_shadow(self.ball)
        
        # Draw ground with camera position for LOD
        self.ground.draw(self.camera.position)
        
        # Draw Obama statue
        self.obama_statue.draw()
        
        # Draw reflective ball
        self.ball.draw(self.env_texture)
        
        # Draw sun rays
        self.draw_sun_rays_screen()
        
        self.draw_hud()
        
        pygame.display.flip()
    
    def run(self):
        print("=" * 50)
        print("3D Engine - Hyperrealistic Silver Ball")
        print("=" * 50)
        print("\nControls:")
        print("  WASD - Move camera")
        print("  Mouse - Look around")
        print("  Space - Move up")
        print("  Shift - Move down")
        print("  Left Click - Push ball")
        print("  F - Apply force to ball")
        print("  R - Reset ball position")
        print("  ESC - Quit")
        print("\nFeatures:")
        print("  - Light wind affecting grass and ball")
        print("  - Uncapped FPS with delta-time physics")
        print("  - Hyperrealistic lighting & shadows")
        print("=" * 50)
        
        while self.running:
            # Calculate delta time for frame-rate independent physics
            current_time = pygame.time.get_ticks() / 1000.0
            delta_time = current_time - self.last_time
            self.last_time = current_time
            
            # Update FPS display
            self.fps_timer += delta_time
            if self.fps_timer >= 0.5:
                self.fps_display = 1.0 / delta_time if delta_time > 0 else 0
                pygame.display.set_caption(f"3D Engine - Hyperrealistic | FPS: {int(self.fps_display)}")
                self.fps_timer = 0
            
            self.handle_events()
            self.update(delta_time)
            self.render()
            
            # Don't limit FPS - let it run as fast as possible
            self.clock.tick()  # No argument = no limit
        
        pygame.quit()


if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
