// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#if PRECISION == 3232 || PRECISION == 6464

INLINE_FUNC real zsqrt(real a) {
  real z;
  singlereal m = hypot(a.x, a.y);
  z.x = sqrt((m + a.x) / 2);
  z.y = copysign(sqrt((m - a.x) / 2), a.y);
  return z;
}

INLINE_FUNC real zexp(real a) {
  real z;
  singlereal e = exp(a.x);
  z.x = e * cos(a.y);
  z.y = e * sin(a.y);
  return z;
}

INLINE_FUNC real zlog(real a) {
  real z;
  z.x = log(hypot(a.x, a.y));
  z.y = atan2(a.y, a.x);
  return z;
}

INLINE_FUNC real zpow(real a, real b) {
  real z;
  Multiply(z, b, zlog(a));
  return zexp(z);
}

INLINE_FUNC real __sqr(real a) {
  real z;
  z.x = (a.x - a.y) * (a.x + a.y);
  z.y = 2 * a.x * a.y;
  return z;
}

INLINE_FUNC real zasinh(real a) {
  real z = __sqr(a);
  z.x += ONE;
  z = zsqrt(z);
  z.x += a.x;
  z.y += a.y;
  z = zlog(z);
  z.x = copysign(z.x, a.x);
  z.y = copysign(z.y, a.y);
  return z;
}

INLINE_FUNC real zacosh(real a) {
  real z = __sqr(a);
  z.x -= ONE;
  z = zsqrt(z);
  z.x += a.x;
  z.y += a.y;
  z = zlog(z);
  z.x = copysign(z.x, ZERO);
  z.y = copysign(z.y, a.y);
  return z;
}

INLINE_FUNC real zatanh(real a) {
  real z;
  real u = a, v = a;
  u.x = ONE + u.x;
  v.x = ONE - v.x; v.y = -v.y;
  DivideFull(z, u, v);
  z = zlog(z);
  z.x /= 2; z.y /= 2;
  z.x = copysign(z.x, a.x);
  z.y = copysign(z.y, a.y);
  return z;
}

INLINE_FUNC real zsinh(real a) {
  real z;
  z.x = sinh(a.x) * cos(a.y);
  z.y = cosh(a.x) * sin(a.y);
  return z;
}

INLINE_FUNC real zcosh(real a) {
  real z;
  z.x = cosh(a.x) * cos(a.y);
  z.y = sinh(a.x) * sin(a.y);
  return z;
}

INLINE_FUNC real ztanh(real a) {
  real z;
  singlereal _2r = 2 * a.x;
  singlereal _2i = 2 * a.y;
  singlereal _d = cosh(_2r) + cos(_2i);
  z.x = sinh(_2r) / _d;
  z.y = sin(_2i) / _d;
  return z;
}

INLINE_FUNC real zasin(real a) {
  real z, t;
  t.x = -a.y; t.y = a.x;
  t = zasinh(t);
  z.x = t.y; z.y = -t.x;
  return z;
}

INLINE_FUNC real zacos(real a) {
  real z;
  real t = __sqr(a);
  t.x -= ONE;
  t = zsqrt(t);
  t.x = a.x + t.x;
  t.y = a.y + t.y;
  t = zlog(t);
  z.x = fabs(t.y);
  z.y = fabs(t.x);
  if (!signbit(a.y))
    z.y = -z.y;
  return z;
}

INLINE_FUNC real zatan(real a) {
  real z, t;
  t.x = -a.y; t.y = a.x;
  t = zatanh(t);
  z.x = t.y; z.y = -t.x;
  return z;
}

INLINE_FUNC real zsin(real a) {
  real z, t;
  t.x = -a.y; t.y = a.x;
  t = zsinh(t);
  z.x = t.y; z.y = -t.x;
  return z;
}

INLINE_FUNC real zcos(real a) {
  real z;
  z.x = -a.y; z.y = a.x;
  return zcosh(z);
}

INLINE_FUNC real ztan(real a) {
  real z, t;
  t.x = -a.y; t.y = a.x;
  t = ztanh(t);
  z.x = t.y; z.y = -t.x;
  return z;
}

#endif

)" // End of the C++11 raw string literal
