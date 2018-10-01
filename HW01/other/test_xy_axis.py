import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
print(x)

x_line= 0
y1 = 2*x + 1
y2 = x**2

y=0

plt.figure()
plt.plot(x, y2)
plt.plot( y)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.show()
