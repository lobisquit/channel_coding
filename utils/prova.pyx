from libc.math cimport sqrt

def is_prime_c(long n):
    cdef long k = 2
    u = sqrt(n)
    while k <= u:
        if n % k == 0:
            return False
        k += 1

    return True
