# Observation
- When trying to integrating with the direct version of euler backward, we observe a singularity at theta - pi/2.  

# Local truncation error : 
Reminder : scheme order = global error = for e forwareed euler at the end of the integration :

$$y_{t_n} - y_n = y(t_n) - (y_0 + hf(y_0) + hf(y_1) + hf(y_2) \dots)$$

Local truncation error is the error commited on one step. 

It is one order higher than the global error (= scheme order). 

Noting $y(t_{n+1})$ the true solution and $y_{n+1}$ the estimation obtained by the scheme, the Local Truncation Error is : 

$$ LTE = y(t_{n+1}) - y_{n+1} \sim O(h^{p+1})$$ 
where p is the scheme order.
### Estimation of the LTE to compute the convergence criterium in the Newton of functions euler_backward_iterative and midpoint :  
#### Midpoint :
- using the "global" error on one Euler step : 
$$hf(y_n) $$ 

as estimation of $O(h)$, we search an estimation of the $O(h^3)$ error form the midpoint.

- the total is scaled 
  - by $h^2$ to ensure $LTE \sim h^3$,
  - by $c=0.001$ to ensure Newton error << integration scheme error.

Resulting in a final tolerance criterium expression :
$$tol = c h^3 \| f(t,y_n)\| $$

#### Euler :
$$tol = c h^2 \| f(t,y_n)\| $$

# note 
the notion of torsor comes form the fact that we try to apply a force vector to a position vector in the form :   

$$rF$$

the first part is an inner product the second an outer

i in dimension 2 and i,j,k in dimesnsion 3

you look for a 2nd / imaginary part that is a rotation , namely an isometry that conserves distance or the absolute value :

$$|u(v-w)| = |u||u-w|$$

in other words, it has to leave O inplace and to conserve distances. 

the cross product of two vector is a pseudo vector, the wedge poduct is an area bivector .i.e a bilinear form that conserves norms. 

1. bilinearity
2. SO(2) invariance $M(e^{i\theta}r,e^{i\theta}F) = M(r,F)$
3. orientation sensitiviy $M(r,F) = -M(r,F)$ under reflection

#### Or the unique bilinear, rotation-equivariant, orientation-sensitive map form $\mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$


where the cross product outputs a vector , the wedge product is a bilinear form. 

the rotation is equivalent to 2 consecutive reflections about two non // lines.  

the translation is equivalent to 2 consecutive reflections about two // lines.

the wedge product is called the fundamental form. 
$$ \begin{pmatrix} i \ j \ k \\ q_1 \ q_2 \ q_3 \\ p_1 \ p_2 \ p_3\end{pmatrix}$$

with 
$$i^2 = j^2 = k^2 = ijk = -1$$
which impli

# why i? 
$$z \in SO(2) , z = \cos(\theta) + i \sin(\theta) $$ 
$$ $$



