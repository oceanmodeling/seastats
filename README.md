# SeaStats

Simple package to compare and anylyse 2 time series. We use the following conventionn in this repo: 
 * `sim`: modelled surge time series
 * `mod`: observed surge time series

## Metrics available: 

These are following metrics available in this repository: 

We need metrics to assess the quality of the model.
We define the most important ones, as stated on this [Stats wiki](https://cirpwiki.info/wiki/Statistics):
### A. Dimensional Statistics:
#### Mean Error (or Bias)
$$\langle x_c - x_m \rangle = \langle x_c \rangle - \langle x_m \rangle$$
#### RMSE (Root Mean Squared Error)
$$\sqrt{\langle(x_c - x_m)^2\rangle}$$
#### Mean-Absolute Error (MAE):
$$\langle |x_c - x_m| \rangle$$
### B. Dimentionless Statistics (best closer to 1)

#### Performance Scores (PS) or Nash-Sutcliffe Eff (NSE): $$1 - \frac{\langle (x_c - x_m)^2 \rangle}{\langle (x_m - x_R)^2 \rangle}$$
#### Correlation Coefficient (R):
$$\frac {\langle x_{m}x_{c}\rangle -\langle x_{m}\rangle \langle x_{c}\rangle }{{\sqrt {\langle x_{m}^{2}\rangle -\langle x_{m}\rangle ^{2}}}{\sqrt {\langle x_{c}^{2}\rangle -\langle x_{c}\rangle ^{2}}}}$$
#### Kling–Gupta Efficiency (KGE):
$$1 - \sqrt{(r-1)^2 + b^2 + (g-1)^2}$$
with :
 * `r` the correlation
 * `b` the modified bias term (see [ref](https://journals.ametsoc.org/view/journals/clim/34/16/JCLI-D-21-0067.1.xml)) $$\frac{\langle x_c \rangle - \langle x_m \rangle}{\sigma_m}$$
 * `g` the std dev term $$\frac{\sigma_c}{\sigma_m}$$

#### Lambda index ($\lambda$), values closer to 1 indicate better agreement:
$$\lambda = 1 - \frac{\sum{(x_c - x_m)^2}}{\sum{(x_m - \overline{x}_m)^2} + \sum{(x_c - \overline{x}_c)^2} + n(\overline{x}_m - \overline{x}_c)^2 + \kappa}$$
 * with `kappa` $$2 \cdot \left| \sum{((x_m - \overline{x}_m) \cdot (x_c - \overline{x}_c))} \right|$$

### ADD STORM SELECTION PROCESS
from methodology explained in this [notebook](https://tomsail.github.io/static/Model_metrics.html)
### ADD "Slope" parameters
more info can be found [in this preprint](https://doi.org/10.5194/egusphere-2024-1415) from Campos-Caba et. al.
"""