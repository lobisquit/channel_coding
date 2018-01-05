import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from LDPC import phi_tilde_vector

x = np.logspace(-6, 2, num=100)
y = phi_tilde_vector(x)

plt.style.use('dark_background')
plt.figure(0, figsize=(5, 3.1))
plt.semilogx(x, y, color='#ff6e64')
plt.grid(linestyle='--', linewidth=0.5)
plt.ylabel('y', rotation=0)
plt.xlabel('x')
plt.tight_layout()
plt.savefig('report/figures/phi_tilde.pdf', transparent=True)
plt.show()
