# Method of moments for antennas
The method of moments is a numerical technique used to approximately solve differential, integral and integro-differential equations. The main idea is to expand an unknown function as a linear combination of basis functions:
$$ g = \sum_{i=1}^{n}a_i L f_i $$
The numerical solution will be accurate if the residual is equal to zero. To minimize the residual, we introduce weight functions and use the Galerkin's method (weight and basis functions are the same). Scalar multiplication on weight functions is equal to zero:
$$ R = g - \sum_{i=1}^{n}a_i L f_i \Rightarrow \left( \omega_{j} \cdot \sum_{i=1}^{n}a_i L(f_i) \right)- \left( \omega_{j} \cdot g \right) = 0$$
From here we get a system of linear equations $m$ by $n$, where $m$ is the number of weight functions, and $n$ is the number of basic ones. 

In case of calculating directional pattern of antenna we can solve Pocklington's equation:
$$j\omega\mu \left( 1 + \dfrac{1}{k^2} \dfrac{\partial^2}{{\partial z}^2}\right) \int_{V} I(z) \dfrac{exp(-jkR)}{4\pi R} \,dz \ = E_z^i$$
Method of moments converts functional equation into a matrix equation:
$$ ZI = V$$
where $Z$ - impedance matrix, $I$ - current matrix, V - operating voltage

or Hallen's equation:
$$j\omega\mu \int_{V} I(z) \left( 1 + \dfrac{1}{k^2} \dfrac{\partial^2}{{\partial z}^2}\right)\dfrac{exp(-jkR)}{4\pi R} \,dz \ = E_z^i$$
Method of moments converts functional equation into a matrix equation:
$$ ZI = C_1s_1+C_2s_2 + b$$
where $Z$ - block impedance matrix, $I$ - block current, s_1 and s_2 - homogeneous solution, b - nonhomogeneous part

