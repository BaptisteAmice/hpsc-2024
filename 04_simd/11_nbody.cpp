#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    //vectorized version
    __m512 x_vec = _mm512_load_ps(x);
    __m512 y_vec = _mm512_load_ps(y);
    __m512 m_vec = _mm512_load_ps(m);

    //float rx = x[i] - x[j];
    __m512 rx_vec = _mm512_sub_ps(_mm512_set1_ps(x[i]), x_vec);
    //float ry = y[i] - y[j];
    __m512 ry_vec = _mm512_sub_ps(_mm512_set1_ps(y[i]), y_vec);

    //float r = std::sqrt(rx * rx + ry * ry);
    __m512 rx_vec2 = _mm512_mul_ps(rx_vec, rx_vec);
    __m512 ry_vec2 = _mm512_mul_ps(ry_vec, ry_vec);
    __m512 r_vec2 = _mm512_add_ps(rx_vec2, ry_vec2);
    __m512 r_vec = _mm512_rsqrt14_ps(r_vec2);

    //fx[i] -= rx * m[j] / (r * r * r);
    //fy[i] -= ry * m[j] / (r * r * r);
    __m512 r2_vec = _mm512_mul_ps(r_vec, r_vec);
    __m512 r3_vec = _mm512_mul_ps(r2_vec, r_vec);
    
    __m512 fx_vec = _mm512_mul_ps(rx_vec, r3_vec);
    fx_vec = _mm512_mul_ps(fx_vec, m_vec);
    
    __m512 fy_vec = _mm512_mul_ps(ry_vec, r3_vec);
    fy_vec = _mm512_mul_ps(fy_vec, m_vec);

    //mask if(i != j)
    __m512 fx_mask = _mm512_setzero_ps();
    __m512 fy_mask = _mm512_setzero_ps();
    __m512 limit = _mm512_set1_ps(10e+5);
    
    __mmask16 mask_x = _mm512_cmp_ps_mask(fx_vec, limit, _MM_CMPINT_GT);
    fx_mask = _mm512_mask_blend_ps(mask_x, fx_vec, fx_mask);
    __mmask16 mask_y = _mm512_cmp_ps_mask(fy_vec, limit, _MM_CMPINT_GT);
    fy_mask = _mm512_mask_blend_ps(mask_y, fy_vec, fy_mask);
    
    //fx[i] -= rx * m[j] / (r * r * r);
    fx[i] -= _mm512_reduce_add_ps(fx_mask);
    //fy[i] -= ry * m[j] / (r * r * r);
    fy[i] -= _mm512_reduce_add_ps(fy_mask);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
