# Метод Моментов (МоМ) для произвольно-ориентированной антенны из тонких проволок

## Electric field integral equation (EFIE) в аппроксимации тонких проводов

Поле, генерируемое распределением токов $\mathbf{J}(\mathbf{r}')$, можно найти из уравнения на векторный потенциал:

$$
\mathbf{E}(\mathbf{r}) = -j\omega \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] \mathbf{A}(\mathbf{r})
$$

Где векторный потенциал определяется следующим образом:

$$
\mathbf{A}(\mathbf{r}) = \mu \int_V \mathbf{J}(\mathbf{r}') G(\mathbf{r}, \mathbf{r}') \mathrm{d}^3 r', \quad \text{здесь} \quad G(\mathbf{r}, \mathbf{r}') = \frac{e^{-jk |\mathbf{r} - \mathbf{r}'|}}{4\pi |\mathbf{r} - \mathbf{r}'|}
$$

Рассмотрим антенну из прямолинейных тонких проволок, в таком приближении векторный потенциал можно записать так:

$$
\mathbf{A}(\mathbf{r}) = \frac{\mu}{4\pi} \int_C \mathbf{I}(\mathbf{r}') \frac{e^{-jk|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|} \mathrm{d}r'
$$

Тогда итоговое уравнение на излучаемое поле примет вид:

$$
\mathbf{E}(\mathbf{r}) = -j \omega \mu \int_C \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] \mathbf{I}(\mathbf{r}') G(\mathbf{r}, \mathbf{r}') \mathrm{d} r'
$$

Так как поверхность проводника эквипотенциальна, касательная компонента поля должна удовлетворять следующему граничному условию:

$$
E_{\mathrm{tan}}(\mathbf{r}) + E_{\mathrm{tan}}^{\mathrm{inc}}(\mathbf{r}) = 0, \quad \text{где} \quad E_{\mathrm{tan}}^{\mathrm{inc}} \text{ — падающее поле (создаваемое источником антенны)}
$$

Скалярно домножим на направляющий вектор $\mathbf{\tau}(\mathbf{r})$ и заменим излучаемое поле на падающее:

$$
E_{\mathrm{tan}}^{\mathrm{inc}}(\mathbf{r}) = \frac{j \omega \mu}{4 \pi} \mathbf{\tau}(\mathbf{r}) \cdot \int_C \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] \mathbf{I}(\mathbf{r}') \frac{e^{-jk|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|} \mathrm{d} r'
$$

## Дискретизация задачи, МоМ и метод Галеркина

Дискретизируем задачу, разбив антенну на цилиндрические сегменты — элементы МоМа. Распределение тока представим как линейную комбинацию базисных функций:

$$
\mathbf{I}(\mathbf{r}') = \sum_{n=0}^N I_n f_n \mathbf{\tau}(\mathbf{r}')
$$

Тогда исходное уравнение примет вид:

$$
E_{\mathrm{tan}}^{\mathrm{inc}}(\mathbf{r}) = \frac{j \omega \mu}{4 \pi} \mathbf{\tau}(\mathbf{r}) \cdot \sum_{n=0}^N I_n \int_{f_n} f_n \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] \mathbf{\tau}(\mathbf{r}') \frac{e^{-jk|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|} \mathrm{d} r'
$$

Введем набор весовых функций $f_m$, аналогичный базисным, и запишем их свертку с левой и правой частями уравнения:

$$
\int_{f_m} E^{\mathrm{inc}}_{\mathrm{tan}} \mathrm{d} r = \frac{j \omega \mu}{4 \pi} \sum_{n=0}^N I_n \int_{f_m} \int_{f_n} f_m f_n \mathbf{\tau}(\mathbf{r}) \cdot \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] \mathbf{\tau}(\mathbf{r}') \frac{e^{-jk|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|} \mathrm{d} r' \mathrm{d} r
$$

Для удобства введем обозначения:

$$
\begin{cases}
\mathbf{r} = \mathbf{r}_m \\
\mathbf{r}' = \mathbf{r}_n \\
\mathbf{\tau}(\mathbf{r}) = \mathbf{\tau}_m \\
\mathbf{\tau}(\mathbf{r}') = \mathbf{\tau}_n \\
|\mathbf{r}_m - \mathbf{r}_n| = R \\
G = \frac{e^{-jkR}}{R}
\end{cases}
$$

В такой нотации уравнение будет выглядеть следующим образом:

$$
\int_{f_m} E^{\mathrm{inc}}_{\mathrm{tan}} \mathrm{d} r_m = \frac{j \omega \mu}{4\pi} \sum_{n=0}^N I_n \int_{f_m} \int_{f_n} f_m f_n \mathbf{\tau}_m \cdot \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] G \mathbf{\tau}_n \mathrm{d} r_n \mathrm{d} r_m
$$

## Построение матричного уравнения

Метод Моментов сводит интегральное уравнение к матричному виду $\mathbf{\overline{Z}I = V}$ размера $M \times M$, где $M$ — количество прямолинейных элементов антенны:

$$
\begin{pmatrix}
\mathbf{Z}_{00} & \mathbf{Z}_{01} & \cdots & \mathbf{Z}_{0M} \\
\mathbf{Z}_{10} & \ddots          & \ddots & \vdots \\
\vdots          & \ddots          & \ddots & \vdots \\
\mathbf{Z}_{M0} & \mathbf{Z}_{M1} & \cdots & \mathbf{Z}_{MM}
\end{pmatrix}
\begin{pmatrix}
\mathbf{I}_0 \\
\mathbf{I}_1 \\
\vdots \\
\mathbf{I}_M
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{V}_0 \\
\mathbf{V}_1 \\
\vdots \\
\mathbf{V}_M
\end{pmatrix}
$$

Здесь напряжение на $m$-ом сегменте $i$-ой палки можно найти следующим образом:

$$
V_{im} = \int_{f_m} E^{\mathrm{inc}}_{\mathrm{tan}} \mathrm{d} r_m
$$

А импеданс:

$$
Z_{ijmn} = \frac{j \omega \mu}{4 \pi} \int_{f_m} \int_{f_n} f_m f_n \mathbf{\tau}_m \cdot \left[ 1 + \frac{1}{k^2} \nabla \nabla \cdot \right] G \mathbf{\tau}_n \mathrm{d} r_n \mathrm{d} r_m
$$

Решением слау являются амплитуды базисных функций, из которых легко находится распределение токов и далее рассчитывается излучение антенны.

## Нахождение напряжений (точечный источник)

Модель точечного источника предполагает дельта-образную функцию падающего поля на одном сегменте МоМа:

$$
V_{im} = \int_{f_m} E^{\mathrm{inc}}_{\mathrm{tan}} \mathrm{d} r_m = \int_{f_m} \frac{V_0}{\Delta r} \delta(r_m - r_{\mathrm{source}}) \mathrm{d} r_m = V_0, \quad \text{если} \ r_{\mathrm{source}} \in f_m
$$

В остальных случаях $V_{im} = 0$.

## Нахождение импеданса

Направляющие вектора определяются следующим образом через зенитный и азимутальный углы:

$$
\begin{cases}
\mathbf{\tau}_m = \sin \phi_m \cos \theta_m \mathbf{e}_x + \sin \phi_m \sin \theta_m \mathbf{e}_y + \cos \phi_m \mathbf{e}_z \\
\mathbf{\tau}_n = \sin \phi_n \cos \theta_n \mathbf{e}_x + \sin \phi_n \sin \theta_n \mathbf{e}_y + \cos \phi_n \mathbf{e}_z
\end{cases}
$$

Теперь попробуем преобразовать подынтегральное выражение:

$$
\left[1 + \frac{1}{k^2} \nabla \nabla \cdot \right] G \mathbf{\tau}_n = G \mathbf{\tau}_n + \frac{1}{k^2} \nabla \left( \tau_{nx} \frac{\partial G}{\partial x} + \tau_{ny} \frac{\partial G}{\partial y} + \tau_{nz} \frac{\partial G}{\partial z} \right) = G\mathbf{\tau_n} + \dfrac{1}{k^2}\begin{pmatrix}
\tau_{nx} \dfrac{\partial^2 G}{\partial x^2} + \tau_{ny} \dfrac{\partial^2 G}{\partial x \partial y} + \tau_{nz} \dfrac{\partial^2 G}{\partial x \partial z} \\
 \tau_{ny} \dfrac{\partial^2 G}{\partial y^2} + \tau_{nx} \dfrac{\partial^2 G}{\partial y \partial x} + \tau_{nz} \dfrac{\partial^2 G}{\partial y \partial z} \\
\tau_{nz} \dfrac{\partial^2 G}{\partial z^2} + \tau_{nx} \dfrac{\partial^2 G}{\partial z \partial x} + \tau_{ny} \dfrac{\partial^2 G}{\partial z \partial y}
\end{pmatrix}
$$


Далее скалярно умножим полученные вектора на $\mathbf{\tau_m}$:
$$
G\mathbf{\tau_m}\cdot\mathbf{\tau_n} + \dfrac{1}{k^2}\tau_{mx}\tau_{nx}\dfrac{\partial^2 G}{\partial x^2} + \dfrac{1}{k^2}\tau_{my}\tau_{ny}\dfrac{\partial^2 G}{\partial y^2} + \dfrac{1}{k^2}\tau_{mz}\tau_{nz}\dfrac{\partial^2 G}{\partial z^2} + \\ + \dfrac{1}{k^2}\left( \tau_{mx}\tau_{ny} + \tau_{my}\tau_{nx}\right) \dfrac{\partial^2 G}{\partial x \partial y} + \dfrac{1}{k^2}\left( \tau_{mx}\tau_{nz} + \tau_{mz}\tau_{nx}\right) \dfrac{\partial^2 G}{\partial x  \partial z} + \dfrac{1}{k^2}\left( \tau_{my}\tau_{nz} + \tau_{mz}\tau_{ny}\right) \dfrac{\partial^2 G}{\partial y \partial z}
$$
Сдеаем несколько замен:

$$
\begin{cases}
c_0 = \mathbf{\tau}_m \cdot \mathbf{\tau}_n \\
c_{xx} = \frac{1}{k^2} \tau_{mx} \tau_{nx} \\
c_{yy} = \frac{1}{k^2} \tau_{my} \tau_{ny} \\
c_{zz} = \frac{1}{k^2} \tau_{mz} \tau_{nz} \\
c_{xy} = \frac{1}{k^2} ( \tau_{mx} \tau_{ny} + \tau_{my} \tau_{nx} ) \\
c_{xz} = \frac{1}{k^2} ( \tau_{mx} \tau_{nz} + \tau_{mz} \tau_{nx} ) \\
c_{yz} = \frac{1}{k^2} ( \tau_{my} \tau_{nz} + \tau_{mz} \tau_{ny} )
\end{cases}
$$

Выражения для импеданса примет следующий вид:

$$
Z_{ijmn} = \frac{j \omega \mu}{4 \pi} \int_{f_m} \int_{f_n} f_m f_n \left[ c_0 G + c_{xx} \frac{\partial^2 G}{\partial x^2} + c_{yy} \frac{\partial^2 G}{\partial y^2} + c_{zz} \frac{\partial^2 G}{\partial z^2} + c_{xy} \frac{\partial^2 G}{\partial x \partial y} + c_{xz} \frac{\partial^2 G}{\partial x \partial z} + c_{yz} \frac{\partial^2 G}{\partial y \partial z} \right] \mathrm{d} r_n \mathrm{d} r_m
$$
Найдем вторую производную $G$, но, чтобы не делать одну и ту же работу дважды, напишем ее для произвольных $q_i,q_j \in \{ x,y,z\}$
$$
\dfrac{\partial}{\partial q_i} \dfrac{\partial}{\partial q_j} G = \dfrac{\partial}{\partial q_i} \dfrac{\partial}{\partial q_j}\dfrac{e^{-jkR}}{R} = \dfrac{\partial}{\partial q_i} \left[ -jk e^{-jkR}\dfrac{1}{R} \dfrac{\partial R}{\partial q_j} - \dfrac{1} {R^2} e^{-jkR}\dfrac{\partial R}{\partial q_j}\right]  =  -\dfrac{\partial}{\partial q_i} \left[ \dfrac{\partial R}{\partial q_j}\dfrac{1+jkR}{R^2}e^{-jkR} \right]=
$$
$$
= -\dfrac{\partial^2 R}{\partial q_i \partial q_j} \dfrac{1+jkR}{R^2}e^{-jkR}-\dfrac{\partial R}{\partial q_j} \dfrac{jk\dfrac{\partial R}{\partial q_i} R^2 - \left(1+jkR\right)\cdot2R\dfrac{\partial R}{\partial q_i}}{R^4}e^{-jkR} + \dfrac{\partial R}{\partial q_j} \dfrac{1+jkR}{R^2} jk\dfrac{\partial R}{\partial q_i} e^{-jkR} =
$$
$$
= e^{-jkR} \left[ -\dfrac{\partial^2 R}{\partial q_i \partial q_j} \dfrac{1+jkR}{R^2} + \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{2+jkR}{R^3} + \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{jk-k^2R}{R^2}\right]
$$
Теперь заметим, что:
$$
\dfrac{\partial ^2 R}{\partial q_i \partial q_j} = \dfrac{\partial}{\partial q_i} \left( \dfrac{q_j - q_j'}{R}\right) = \dfrac{\delta_{ij}}{R} - \dfrac{\Delta q_j}{R^2} \dfrac{\partial R}{\partial q_i} = \dfrac{\delta_{ij}}{R} - \dfrac{1}{R} \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j}, \space \text{где} \ \delta_{ij} - \text{символ Кронекера}
$$
Подставим это замечание и продолжим преобразования:
$$
e^{-jkR} \left[ -\dfrac{\delta_{ij}}{R}\dfrac{1+jkR}{R^2} + \dfrac{1}{R}\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{1+jkR}{R^2} + \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{2+jkR}{R^3} + \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{jk-k^2R}{R^2}\right]
 = 
$$
$$
=e^{-jkR} \left[ -\dfrac{1+jkR}{R^3}\delta_{ij} +\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j}\dfrac{1+jkR+2+jkR+jkR-k^2R^2}{R^3}\right] = 
$$
$$
= e^{-jkR}\left[ \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \dfrac{3+3jkR-k^2R^2}{R^3}-\delta_{ij}\dfrac{1+jkR}{R^3}\right] = \dfrac{e^{-jkR}}{R^3} \left[ \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \left(3+3jkR-k^2R^2\right)
-\delta_{ij} \left(1+jkR\right) \right]
$$
Теперь выделим мнимую и вещественную части:
$$
 \dfrac{e^{-jkR}}{R^3} \left[ \dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \left(3+3jkR-k^2R^2\right)
-\delta_{ij} \left(1+jkR\right) \right] = \dfrac{e^{-jkR}}{R^3} \left[\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \left({3-k^2R^2}\right) - \delta_{ij} +jkR\left(3\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j}-\delta_{ij}\right)\right]
$$
Вернемся к импедансу, подынтегральное выражение (без базисных и весовых функций) теперь можно записать в следующем виде:
$$
c_0 G + \sum_{q_i} \sum_{q_j} c_{q_i q_j} \dfrac{\partial^2 G}{\partial q_i \partial q_j} = c_0 G + \sum_{q_i} \sum_{q_j} c_{q_i q_j} \dfrac{e^{-jkR}}{R^3}\left[\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \left({3-k^2R^2}\right) - \delta_{ij} +jkR\left(3\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j}-\delta_{ij}\right)\right] =
$$
$$ 
= c_0\dfrac{e^{-jkR}}{R} + \dfrac{e^{-jkR}}{R^3}\sum_{q_i} \sum_{q_j} c_{q_i q_j}\left[\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} \left(3-k^2R^2\right) - \delta_{ij} +jkR\left(3\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j}-\delta_{ij}\right)\right] =
$$
$$
= \dfrac{e^{-jkR}}{R^3}\left( c_0R^2 + \sum_{q_i} \sum_{q_j} c_{q_i q_j}\left[\Delta q_i \Delta q_j \dfrac{\left(3-k^2R^2\right)}{R^2} - \delta_{ij} +jkR\left(\dfrac{3}{R^2}\Delta q_i \Delta q_j -\delta_{ij}\right)\right] \right) =
$$
$$
= \dfrac{e^{-jkR}}{R^3} \left[ c_0R^2 + \dfrac{3-k^2R^2}{R^2} \sum_{q_i} \sum_{q_j} c_{q_i q_j}\Delta q_i \Delta q_j - \sum_{q_i} \sum_{q_j} c_{q_i q_j}\delta_{ij} + \dfrac{3jk}{R}\sum_{q_i} \sum_{q_j} c_{q_i q_j}\Delta q_i \Delta q_j -jkR\sum_{q_i} \sum_{q_j} c_{q_i q_j}\delta_{ij}\right]=
$$
Введем следующие замены для удобства:
$$
\begin{cases}
C =  \sum_{q_i} \sum_{q_j} c_{q_i q_j} \delta_{ij} = c_{xx}+c_{yy}+c_{zz} \\
L = \sum_{q_i} \sum_{q_j} c_{q_i q_j}\Delta q_i \Delta q_j = c_{xx}\Delta x^2 + c_{yy}\Delta y^2 + c_{zz} \Delta z^2 + c_{xy}\Delta x \Delta y + c_{xz} \Delta x \Delta z + c_{yz} \Delta y \Delta z
\end{cases}
$$

В итоге в выражении для импеданса получим:

$$
Z_{ijmn} = \frac{j \omega \mu}{4 \pi} \int_{f_m} \int_{f_n} f_m f_n \frac{e^{-jkR}}{R^3} \left[ \left( c_0 R^2 + \frac{3 - k^2 R^2}{R^2} L - C \right) + j k \left( \frac{3L}{R} - R C \right) \right] \mathrm{d} r_n \mathrm{d} r_m
$$

Представляя $e^{-jkR} = \cos kR - j \sin kR$выделим мнимую и вещественную части импеданса, введя следующие обозначения:

$$
\begin{cases}
P_{\mathrm{real}} = c_0 R^2 - C + \frac{3 - k^2 R^2}{R^2} L \\
P_{\mathrm{imag}} = k \left( \frac{3L}{R} - R C \right)
\end{cases}
$$

Тогда

$$
\frac{\cos kR - j \sin kR}{R^3} (P_{\mathrm{real}} + j P_{\mathrm{imag}}) = \left( P_{\mathrm{real}} \frac{\cos kR}{R^3} + P_{\mathrm{imag}} \frac{\sin kR}{R^3} \right) + j \left( P_{\mathrm{imag}} \frac{\cos kR}{R^3} - P_{\mathrm{real}} \frac{\sin kR}{R^3} \right)
$$

Для численного расчета параметризуем кривую, вдоль которой интегрируем:

$$
\begin{cases}
\mathbf{r} = \mathbf{r}_m - \frac{\Delta \mathbf{r}_m}{2} + t_m \Delta \mathbf{r}_m \\
\mathbf{r}' = \mathbf{r}_n - \frac{\Delta \mathbf{r}_n}{2} + t_n \Delta \mathbf{r}_n
\end{cases}
$$

Итоговое выражение:

$$  
Z_{ijmn} = \frac{j \omega \mu \Delta r^2}{4 \pi} \int_{f_m} \int_{f_n} f_m f_n \left[ \left( P_{\mathrm{real}} \frac{\cos kR}{R^3} + P_{\mathrm{imag}} \frac{\sin kR}{R^3} \right) + j \left( P_{\mathrm{imag}} \frac{\cos kR}{R^3} - P_{\mathrm{real}} \frac{\sin kR}{R^3} \right) \right] \mathrm{d} t_n \mathrm{d} t_m
$$

## Self-импеданс
Проведем аналогичные математические выкладки для self-импеданса, то есть формально какой потенциал эффективно наводит элемент МоМа сам на себя. Для этого скажем, что $kR \ll 1$, то есть $e^{-jkR} \approx 1 - jkR$:
$$
G = \dfrac{e^{-jkR}}{R} \approx \dfrac{1}{R}-jk
$$
Тогда 
$$
\dfrac{\partial }{\partial q_i}\dfrac{\partial G}{\partial q_j} =\dfrac{\partial }{\partial q_i} \dfrac{\partial }{\partial q_j} \left(\dfrac{1}{R}-jk\right) =- \dfrac{\partial}{\partial q_i} \left(\dfrac{1}{R^2} \dfrac{\partial R}{\partial q_j} \right)= \dfrac{3}{R^3}\dfrac{\partial R}{\partial q_i}\dfrac{\partial R}{\partial q_j} - \dfrac{\delta_{ij}}{R^3} 
$$
Подынтегральное выражение в принятых ранее обозначениях примет вид:
$$
c_0 \left( \dfrac{1}{R}-jk\right) + \dfrac{1}{R^3}\sum_{q_i}\sum_{q_j}c_{q_i q_j} \left( \dfrac{3 \Delta q_i \Delta q_j}{R^2} - \delta_{ij}\right) = \dfrac{c_0}{R} - \dfrac{C}{R^3} + \dfrac{3L}{R^5}-jkc_0
$$
$$
\begin{cases}
P_{\mathrm{real}}^{\mathrm{self}} = \dfrac{c_0}{R} - \dfrac{C}{R^3} + \dfrac{3L}{R^5} \\
P_{\mathrm{imag}}^{\mathrm{self}} = - kc_0
\end{cases}
$$
В итоге получаем:
$$
Z_{ijmn}^{\mathrm{self}} = \dfrac{j\omega\mu}{4\pi} \int_{f_m}\int_{f_n} f_m f_n \left( P_{\mathrm{real}}^{\mathrm{self}} + jP_{\mathrm{imag}}^{\mathrm{self}}\right) \mathrm{d}r_n \mathrm{d}r_m
$$
После параметризации:
$$
Z_{ijmn}^{\mathrm{self}} = \dfrac{j\omega\mu\Delta r^2}{4\pi} \int_{f_m}\int_{f_n} f_m f_n \left( P_{\mathrm{real}}^{\mathrm{self}} + jP_{\mathrm{imag}}^{\mathrm{self}}\right) \mathrm{d}t_n \mathrm{d}t_m
$$

## Нахождение дальнего поля

Теперь мы можем найти элементы матричного уравнения, а из СЛАУ получить распределение индуцируемых токов. Рассмотрим изначальное уравнение, чтобы найти излучение:
$$
\mathbf{E}\left(\mathbf{r_m}\right) = -\dfrac{j\omega\mu}{4\pi} \int_C \left[ 1 + \dfrac{1}{k^2}\nabla\nabla\cdot\right] \mathbf{I}\left(\mathbf{r'}\right) \dfrac{e^{-jk|\mathbf{r_m}-\mathbf{r'}|}}{|\mathbf{r_m}-\mathbf{r'}|}\mathrm{d}r'
$$
Для дальнего поля пренебрежем слагаемым порядка $\dfrac{1}{|\mathbf{r_m} - \mathbf{r_n}|^2}$ и получим:
$$
E\left(\mathbf{r_m}\right) = -\dfrac{j\omega\mu}{4\pi} \sum_{n=0}^NI_n \int_{f_n} f_n\dfrac{e^{-jk|\mathbf{r_m}-\mathbf{r}'|}}{|\mathbf{r_m}-\mathbf{r_n}|}\mathrm{d}r'
$$
Будем считать, что расстояние, на котором мы считаем излучение сильно больше, чем размер нашей антенны, таким образом норма в знаменателе будет постоянной $R \approx |\mathbf{r_m}-\mathbf{r_n}|$:
$$
E\left(\mathbf{r_m}\right) = -\dfrac{j\omega\mu}{4\pi R} \sum_{n=0}^NI_n \int_{f_n} f_n e^{-jk|\mathbf{r_m}-\mathbf{r}'|}\mathrm{d}r'
$$
Так как $R \gg \dfrac{2D^2}{\lambda}$, можно считать фронт нашей сферической волны плоской:
$$
-jk|\mathbf{r_m} - \mathbf{r'}| = -jk \left(R - \left( \mathbf{r'} - \mathbf{r_n}\right) \cdot \mathbf{\hat k}\right) = -jkR-j \mathbf{k}\cdot\mathbf{r_n} + j\mathbf{k}\cdot\mathbf{r'}
$$
Теперь вынесем получившийся фазовый множитель:
$$
E\left(\mathbf{r_m}\right) = - \sum_{n=0}^N\dfrac{j\omega\mu I_n}{4\pi R} e^{-jkR} e^{-j \mathbf{k}\cdot\mathbf{r_n}}\int_{f_n} f_n e^{j\mathbf{k}\cdot\mathbf{r'}}\mathrm{d}r', \space \text{здесь} \ \mathbf{k} = \dfrac{\omega}{c}\dfrac{\mathbf{r_m}-\mathbf{r_n}}{|\mathbf{r_m}-\mathbf{r_n}|}
$$
Формально, нам нужно найти фурье-образ базисной функции. Пусть она определена на промежутке $\{\mathbf{r_{nmin}};\mathbf{r_{nmax}}\}$. Параметризуем интеграл:
$$
\mathbf{r'} = \mathbf{r_n} - \dfrac{\Delta \mathbf{r_n}}{2} + t_n \Delta \mathbf{r_n}
$$
$$
j\mathbf{k}\cdot\mathbf{r'} = j \left( \mathbf{k}\cdot\mathbf{r_n} - \dfrac{\mathbf{k}\cdot\Delta\mathbf{r_n}}{2} + t_n \mathbf{k}\cdot \Delta \mathbf{r_n}\right) = j\left( \mathbf{k}\cdot\mathbf{r_n} - \dfrac{\mathbf{k}\cdot\Delta\mathbf{r_n}}{2} \right) + j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n
$$
$$
\int_{{t_{\mathrm{nmin}}}}^{{t_{\mathrm{nmax}}}} f_n\Delta r e^{j\left( \mathbf{k}\cdot\mathbf{r_n} -\mathbf{k}\cdot\Delta\mathbf{r_n}/2 \right) + j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n}\mathrm{d}t_n =\Delta r  e^{j\left( \mathbf{k}\cdot\mathbf{r_n} -\mathbf{k}\cdot\Delta\mathbf{r_n}/2 \right)} \int_{{t_{\mathrm{nmin}}}}^{{t_{\mathrm{nmax}}}} f_ne^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n}\mathrm{d}t_n
$$
Итоговое выражение принимает следующий вид:
$$
E\left(\mathbf{r_m}\right) = - \sum_{n=0}^N \dfrac{j\omega\mu I_n\Delta r}{4\pi R} e^{-jkR}e^{-j\mathbf{k}\cdot\Delta\mathbf{r_n}/2} \int_{{t_{\mathrm{nmin}}}}^{{t_{\mathrm{nmax}}}} f_ne^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n}\mathrm{d}t_n
$$
Для импульсной базисной функции $f_n = \mathrm{1}$:
$$
E\left(\mathbf{r_m}\right) = - \sum_{n=0}^N \dfrac{j\omega\mu I_n\Delta r}{4\pi R} e^{-jkR}e^{-j\mathbf{k}\cdot\Delta\mathbf{r_n}/2} \int_0^1e^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n}\mathrm{d}t_n = - \sum_{n=0}^N \dfrac{\omega\mu I_n\Delta r}{4\pi R \mathbf{k}\cdot\Delta\mathbf{r_n}} e^{-jkR}e^{-j\mathbf{k}\cdot\Delta\mathbf{r_n}/2} \left( e^{j\mathbf{k}\cdot\Delta\mathbf{r_n}}-1\right)
$$
Отдельно рассмотрим случай $\mathbf{k}\cdot\Delta\mathbf{r_n} = 0$, для него:
$$
E\left(\mathbf{r_m}\right)  = - \sum_{n=0}^N  \dfrac{j\omega\mu I_n \Delta r}{4 \pi R}e^{-jkR}
$$
Для треугольных базисных функций:
$$
\int_{{t_{\mathrm{nmin}}}}^{{t_{\mathrm{nmax}}}} f_ne^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n}\mathrm{d}t_n = \int_{-1/2}^{1/2} \left( t_n + \dfrac{1}{2} \right)e^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n} \mathrm{d}t_n +  \int_{1/2}^{3/2} \left(\dfrac{3}{2} - t_n\right)e^{ j\mathbf{k}\cdot\Delta \mathbf{r_n}t_n} \mathrm{d}t_n 
$$
Интегрируя по частям получим:
$$
E\left(\mathbf{r_m}\right)= - \sum_{n=0}^N \dfrac{j\omega\mu I_n\Delta r}{4\pi R\left(\mathbf{k}\cdot\Delta\mathbf{r_n}^2\right)} e^{-jkR}e^{-j\mathbf{k}\cdot\Delta\mathbf{r_n}} \left(2e^{j\mathbf{k}\cdot\Delta\mathbf{r_n}} - e^{2j\mathbf{k}\cdot\Delta\mathbf{r_n}}-1\right)
$$
Аналогично при  $\mathbf{k}\cdot\Delta\mathbf{r_n} = 0$:
$$
E\left(\mathbf{r_m}\right)  = - \sum_{n=0}^N  \dfrac{j\omega\mu I_n \Delta r}{2 \pi R}e^{-jkR}
$$

# Программная реализация 
See you soon