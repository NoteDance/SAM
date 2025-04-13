# SAM

**Overview**:

The `SAM` optimizer (Sharpness-Aware Minimization) is designed to improve model generalization by seeking parameters that lie in neighborhoods with uniformly low loss. SAM works by first perturbing the model weights in the direction of the gradients (scaled by a hyperparameter ρ) to explore the sharpness of the loss landscape. It then computes the gradients at these perturbed parameters and finally applies the update using an underlying base optimizer. Optional modifications include adaptive scaling of the perturbation and gradient centralization to further stabilize the update. This two-step process (perturb then update) helps to avoid sharp minima and improves robustness against noise.

**Parameters**:

- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer (e.g., SGD, Adam, AdamW) that performs the parameter updates after the SAM perturbation step.
- **`rho`** *(float, default=0.05)*: The radius defining the neighborhood around the current parameters where the loss is evaluated; it controls the magnitude of the weight perturbation.
- **`adaptive`** *(bool, default=False)*: If True, scales the perturbation by the element-wise absolute value of the weights, leading to adaptive perturbations that account for parameter scale.
- **`use_gc`** *(bool, default=False)*: If True, applies gradient centralization to the gradients (by subtracting the mean) before computing the perturbation.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant added to avoid division by zero when computing the gradient norm for the perturbation step.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options to clip gradients by norm or by value to control gradient explosion.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of model weights, which can help improve stability.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA computation.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in training steps) at which the EMA weights are updated.
- **`loss_scale_factor`** *(float, optional)*: A factor for scaling the loss during gradient computation, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps for which gradients are accumulated before performing an update.
- **`name`** *(str, default="sam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from sam import SAM

# Define a base optimizer (e.g., Adam)
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Wrap the base optimizer with SAM
optimizer = SAM(
    base_optimizer=base_opt,
    rho=0.05,
    adaptive=False,
    use_gc=False,
    perturb_eps=1e-12
)

# Build and compile a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
```

# GSAM

**Overview**:

The `GSAM` optimizer extends the Sharpness-Aware Minimization (SAM) framework by incorporating gradient decomposition into the weight perturbation process. GSAM perturbs the model weights by temporarily zeroing out certain dynamics (e.g., reducing BatchNormalization momentum) and decomposing the gradients into components that are then recombined. This surgery reduces harmful gradient conflicts and encourages convergence toward flatter minima, thereby improving model generalization.

**Parameters**:

- **`model`** *(tf.keras.Model, required)*: The model being optimized. Used for accessing layers (e.g., BatchNormalization) during perturbation.
- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer (e.g., Adam, SGD) that performs the actual weight updates after SAM perturbation.
- **`rho_scheduler`** *(object, required)*: A scheduler that provides the current perturbation radius (rho) at each update.
- **`alpha`** *(float, default=0.4)*: A weighting factor controlling the contribution of gradient decomposition in the SAM update.
- **`adaptive`** *(bool, default=False)*: If True, scales the perturbation adaptively based on weight magnitudes.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant added to avoid division by zero in the perturbation computation.
- **Standard parameters** such as `clipnorm`, `clipvalue`, `global_clipnorm`, `use_ema`, `ema_momentum`, `ema_overwrite_frequency`, `loss_scale_factor`, and `gradient_accumulation_steps` are also available.
- **`name`** *(str, default="gsam")*: Name of the optimizer.

**Example Usage**:
```python
import tensorflow as tf
from sam import GSAM

# Instantiate a base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Define a rho scheduler (example: exponential decay)
rho_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=1000, decay_rate=0.95)

# Instantiate GSAM optimizer
optimizer = GSAM(
    model=model,
    base_optimizer=base_opt,
    rho_scheduler=rho_scheduler,
    alpha=0.4,
    adaptive=False,
    perturb_eps=1e-12
)

# Compile and train the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

---

# WSAM

**Overview**:

The `WSAM` (Weighted SAM) optimizer is a variant of SAM that refines the sharpness-aware update by incorporating a corrective term derived from a decomposition of the gradient. WSAM perturbs the weights and then calculates a corrective gradient based on the differences between the original and perturbed weights. Additionally, it synchronizes BatchNormalization momentum during the update to stabilize training. This two-phase update helps guide the model toward flatter regions of the loss landscape.

**Parameters**:

- **`model`** *(tf.keras.Model, required)*: The model to be optimized; used for synchronizing BatchNormalization layers during perturbation.
- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer that performs the weight update after perturbation.
- **`rho`** *(float, default=0.1)*: Perturbation radius for the SAM update.
- **`k`** *(int, default=10)*: A parameter that controls the frequency of corrective updates in WSAM.
- **`alpha`** *(float, default=0.7)*: Weighting factor for blending the corrective gradient with the original gradient.
- **`adaptive`** *(bool, default=False)*: Whether to use adaptive scaling based on weight magnitudes in the perturbation step.
- **`use_gc`** *(bool, default=False)*: If True, applies Gradient Centralization to the gradients.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant to avoid division by zero in the perturbation computation.
- **Standard parameters** for clipping, EMA, and gradient accumulation are also supported.
- **`name`** *(str, default="wsam")*: Name of the optimizer.

**Example Usage**:
```python
import tensorflow as tf
from sam import WSAM

# Instantiate a base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Instantiate WSAM optimizer
optimizer = WSAM(
    model=model,
    base_optimizer=base_opt,
    rho=0.1,
    k=10,
    alpha=0.7,
    adaptive=False,
    use_gc=False,
    perturb_eps=1e-12
)

# Compile the model using WSAM
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

---

# BSAM

**Overview**:

The `BSAM` (Batch Sharpness-Aware Minimization) optimizer employs a three-step update process designed to further reduce sharpness in the weight space. First, BSAM injects controlled random noise into the weights based on the dataset size. In the second step, it computes an intermediate update by adjusting for the noisy gradient. Finally, a momentum-based correction refines the updates. This layered approach aims to produce more robust and stable training, particularly useful when handling large datasets with noisy gradient estimates.

**Parameters**:

- **`num_data`** *(int, required)*: The total number of training samples; used to scale the injected noise in the first step.
- **`lr`** *(float, default=0.5)*: The learning rate for the momentum update step.
- **`beta1`** *(float, default=0.9)*: Decay rate for the momentum accumulator.
- **`beta2`** *(float, default=0.999)*: Decay rate for the secondary gradient accumulation for variance estimation.
- **`weight_decay`** *(float, default=1e-4)*: Coefficient for weight decay regularization.
- **`rho`** *(float, default=0.05)*: Perturbation strength used during the second update phase.
- **`adaptive`** *(bool, default=False)*: Whether to apply adaptive scaling based on weight magnitudes.
- **`damping`** *(float, default=0.1)*: Damping factor for smoothing the momentum update in the third step.
- **Standard parameters** such as gradient clipping, EMA settings, etc., are also supported.
- **`name`** *(str, default="bsam")*: Name of the optimizer.

**Example Usage**:
```python
import tensorflow as tf
from sam import BSAM

# Instantiate the BSAM optimizer
optimizer = BSAM(
    num_data=50000,
    lr=0.5,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-4,
    rho=0.05,
    adaptive=False,
    damping=0.1
)

# Build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using BSAM
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# LookSAM

**Overview**:  
The `LookSAM` optimizer is an extension of the Sharpness-Aware Minimization (SAM) family that incorporates a lookahead mechanism to improve generalization. It works in two steps. In the first step, if the current step is a multiple of a defined interval k, the optimizer perturbs the weights by adding a scaled version of the gradient (with optional gradient centralization). This perturbation is computed adaptively if desired. In the second step, the optimizer resets the weights to their original values and applies a corrective update based on the difference between the original and perturbed weights, blended with a factor alpha. This two-phase process encourages convergence toward flatter regions of the loss landscape.

**Parameters**:  
- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer (e.g., Adam, SGD) that performs the weight update after LookSAM’s correction.  
- **`rho`** *(float, default=0.1)*: The perturbation radius controlling the magnitude of the weight perturbation.  
- **`k`** *(int, default=10)*: The interval (in steps) at which the lookahead (perturbation and correction) is applied.  
- **`alpha`** *(float, default=0.7)*: The blending factor for the corrective update, determining how much of the correction is applied.  
- **`adaptive`** *(bool, default=False)*: If True, scales the perturbation based on the magnitude of the weights for adaptive adjustments.  
- **`use_gc`** *(bool, default=False)*: If True, applies gradient centralization (subtracting the mean of the gradient) for potentially improved stability.  
- **`perturb_eps`** *(float, default=1e-12)*: A small constant added to avoid numerical instabilities during the perturbation step.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for clipping gradients by their norm, by a fixed value, or using a global norm across all parameters, respectively.  
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.  
- **`ema_momentum`** *(float, default=0.99)*: The momentum coefficient used in EMA calculations if enabled.  
- **`ema_overwrite_frequency`** *(int, optional)*: The frequency (in steps) at which EMA weights are updated.  
- **`loss_scale_factor`** *(float, optional)*: A factor for scaling the loss during gradient computation, useful in mixed-precision training.  
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps over which gradients are accumulated before an update is applied.  
- **`name`** *(str, default="looksam")*: The name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from sam import LookSAM

# Define the base optimizer (e.g., Adam)
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Instantiate LookSAM without a model parameter
optimizer = LookSAM(
    base_optimizer=base_opt,
    rho=0.1,
    k=10,
    alpha=0.7,
    adaptive=False,
    use_gc=False,
    perturb_eps=1e-12
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using LookSAM
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
