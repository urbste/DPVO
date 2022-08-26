import numpy as np

a = 6378137.0
e2 = 6.6943799901377997e-3
a1 = 4.2697672707157535e+4
a2 = 1.8230912546075455e+9
a3 = 1.4291722289812413e+2
a4 = 4.5577281365188637e+9
a5 = 4.2840589930055659e+4
a6 = 9.9330562000986220e-1
f = 1.0 / 298.257223563
e = (2.0 * f - f * f)**0.5
e2 = e * e

def DegToRad(deg):
    return deg / 180.0 * np.pi

def RadToDeg(rad):
    return rad * 180.0 / np.pi

# convert latitudem longitude and altitude to XYZ
def LLAtoECEF(latitude, longitude, altitude):
  lat_rad = DegToRad(latitude)
  lon_rad = DegToRad(longitude)

  n = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) * np.sin(lat_rad))
  x = (n + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
  y = (n + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
  z = (n * (1.0 - e2) + altitude) * np.sin(lat_rad)
  return x,y,z

def ECEFtoENU(ecef, lla0):

  lmbda = DegToRad(lla0[0])
  phi = DegToRad(lla0[1])
  sin_lambda = np.sin(lmbda)
  cos_lambda = np.cos(lmbda)
  sin_phi = np.sin(phi)
  cos_phi = np.cos(phi)

  x0,y0,z0 = LLAtoECEF(lla0[0],lla0[1],lla0[2])

  xd = x0 - ecef[0]
  yd = y0 - ecef[1]
  zd = z0 - ecef[2]

  t = -cos_phi * xd - sin_phi * yd
  enu_x = -sin_phi * xd + cos_phi * yd
  enu_y = t * sin_lambda + cos_lambda * zd
  enu_z = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

  return enu_x, enu_y, enu_z


# def ECEFToLLA(ecef):
#   x = ecef[0]
#   y = ecef[1]
#   z = ecef[2]
#   a2 = a * a

#   w2 = x * x + y * y
#   l = e2 * 0.5
#   l2 = l * l
#   m = w2 / a2
#   n = (z * z) * (1.0 - e2) / a2
#   p = (m + n - 4.0 * l2) / 6.0
#   G = m * n * l2
#   H = 2.0 * p**3 + G

#   C = (H + G + 2.0 * np.sqrt(H * G))**(1./3.) / 2**(1./3.)
#   i = -(2 * l2 + m + n) * 0.5
#   P = p * p
#   beta = i / 3.0 - C - P / C
#   k = l2 * (l2 - m - n)
#   mn = m - n
#   t = np.sqrt(np.sqrt(beta * beta - k) - (beta + i) * 0.5) - np.sign(mn) * np.sqrt(np.abs(beta - i) * 0.5)

#   F = t**4 + 2.0 * i * t * t + 2.0 * l * mn * t + k
#   dFdt = 4.0 * t**3 + 4.0 * i * t + 2.0 * l * mn

#   deltat = -F / dFdt
#   u = t + deltat + l
#   v = t + deltat - l

#   w = np.sqrt(w2)
#   deltaw = w * (1.0 - 1.0 / u)
#   deltaz = z * (1.0 - (1.0 - e2) / v)
#   lat = RadToDeg(np.arctan2(z * u, w * v))
#   lon = RadToDeg(np.arctan2(y, x))
#   alt = np.sign(u - 1.0) * np.sqrt(deltaw * deltaw + deltaz * deltaz)

#   return np.array([lat, lon, alt])

def ECEFToLLA(ecef):
  x = ecef[0]
  y = ecef[1]
  z = ecef[2]

  zp = np.abs(z)
  w2 = x * x + y * y
  w = np.sqrt(w2)
  r2 = w2 + z * z
  r = np.sqrt(r2)
  lon = np.arctan2(y, x)

  s2 = z * z / r2
  c2 = w2 / r2
  u = a2 / r
  v = a3 - a4 / r
  if (c2 > 0.3):
    s = (zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r)
    lat = np.arcsin(s)
    ss = s * s
    c = np.sqrt(1.0 - ss)
  else:
    c = (w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r)
    lat = np.arccos(c)
    ss = 1.0 - c * c
    s = np.sqrt(ss)
  

  g = 1.0 - e2 * ss
  rg = a / np.sqrt(g)
  rf = a6 * rg
  u = w - rg * c
  v = zp - rf * s
  f = c * u + s * v
  m = c * v - s * u
  p = m / (rf / g + f)
  lat = lat + p
  alt = f + m * p / 2.0
  if (z < 0.0):
    lat *= -1.0
  
  return np.array([RadToDeg(lat),RadToDeg(lon), alt])
