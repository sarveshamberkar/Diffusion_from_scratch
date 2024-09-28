# Diffusion

In the context of deep learning and computer vision, diffusion often refers to a specific type of generative model known as **diffusion models**. These models have gained popularity for generating high-quality images and other types of data. Here’s a breakdown of how diffusion models work and their relevance in computer vision:

### What Are Diffusion Models?

1. **Process Overview**:
    - Diffusion models are based on a **two-step process**: the **forward diffusion process** and the **reverse diffusion process**.
    - **Forward Diffusion**: In this phase, noise is gradually added to the data (like images) over several steps until it becomes nearly indistinguishable from pure noise.
    - **Reverse Diffusion**: The model learns to reverse this process, gradually denoising the noisy data to reconstruct the original image.
2. **Training**:
    - During training, the model is trained to predict the original image from a noisy version at various stages of diffusion. This involves minimizing the difference between the model's prediction and the actual original image.
3. **Sampling**:
    - After training, generating new images involves starting with random noise and applying the learned reverse process to gradually refine the noise into a coherent image.

### Advantages of Diffusion Models

- **High Quality**: Diffusion models tend to produce high-fidelity images, often surpassing traditional generative models like GANs (Generative Adversarial Networks) in quality.
- **Stable Training**: They can be easier to train compared to GANs, which can suffer from mode collapse and instability during training.

### Applications in Computer Vision

1. **Image Generation**: Creating realistic images from random noise or text prompts (similar to models like DALL-E).
2. **Image Super-Resolution**: Enhancing the resolution of images by learning to generate high-resolution details from low-resolution inputs.
3. **Inpainting**: Filling in missing or corrupted parts of an image by generating plausible content.

## 1. Sampling

Sampling in the context of diffusion models refers to the process of generating new data (like images) from the model after it has been trained. This process typically involves several steps, and understanding it is crucial for effectively using diffusion models. Here's a detailed breakdown:

### Steps in the Sampling Process

1. **Starting Point**:
    - **Random Noise**: Sampling begins with a tensor of random noise. This noise is typically sampled from a Gaussian distribution.
2. **Reverse Diffusion Process**:
    - The core idea is to iteratively refine this noisy tensor back into a coherent image. This involves a series of steps, each corresponding to a reverse diffusion process.
3. **Iterative Denoising**:
    - The model performs a series of denoising steps, each reducing the noise level.
    - For each step t (from the maximum time step down to 1), the model predicts a less noisy version of the input. t
        
        
    - The denoising is guided by the learned distribution from the training phase, effectively reversing the noise addition.
4. **Mathematical Representation**:
    - Each denoising step can be mathematically represented using a parameterized function (usually a neural network) that predicts the mean and variance of the denoised image at each step.
    - The model takes the noisy image from the previous step and refines it according to its learned parameters:
    xt−1​=μθ​(xt​,t)+σ(t)⋅ϵ
    Here, μθ​ is the model's prediction of the mean, xt​ is the current noisy image, and ϵ is sampled from a normal distribution.
        
        ![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image.png)
        
5. **Stopping Criterion**:
    - This iterative process continues until you reach the last step (usually step 1), where the final output is expected to resemble a sample from the training distribution.

### Considerations in Sampling

- **Step Size**: The number of denoising steps affects the quality of the generated samples. More steps can yield better results but take longer to compute.
- **Temperature**: Adjusting the sampling temperature can affect diversity. A higher temperature may result in more varied outputs, while a lower temperature may yield more similar outputs.
- **Guidance Techniques**: Techniques like classifier-free guidance can enhance the sampling process, allowing for more control over the characteristics of the generated images (e.g., guiding the model to produce images that are more aligned with a specific style or category).

### Applications of Sampling

1. **Image Generation**: Creating entirely new images from scratch.
2. **Image Editing**: Altering images by starting with a noisy version of an existing image and applying targeted modifications.
3. **Text-to-Image Generation**: Using prompts to guide the sampling process to generate images that fit specific descriptions.

PSUDO code for sampling 

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%201.png)

### Classifier-free guidance

Classifier-free guidance is a technique used in diffusion models to improve the quality and relevance of generated samples, particularly in text-to-image generation. It helps control the output of the model to make it more aligned with a given prompt or condition, without needing a separate classifier.

### How Classifier-Free Guidance Works

1. **Two Models**:
    - Instead of using a classifier to steer the generation process, classifier-free guidance leverages a single diffusion model that can generate samples both conditioned on prompts (like text descriptions) and unconditioned (without any prompts).
2. **Training Phase**:
    - During training, the model learns to generate images from noisy inputs based on both conditioned and unconditioned data.
    - It essentially learns two modes:
        - **Conditioned on prompts**: The model generates images that are relevant to a specific input (e.g., "a cat sitting on a mat").
        - **Unconditioned**: The model generates images without any specific prompt.
3. **Guidance During Sampling**:
    - When generating samples, you can control the influence of the conditioning by combining outputs from the conditioned and unconditioned generations.
    - The idea is to interpolate between the two:
        
        
    
    ![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%202.png)
    
    - Here, λ is a hyperparameter that controls the strength of the guidance. A higher λ places more emphasis on the conditioning, resulting in outputs that more closely adhere to the prompt.
        
        λ lambda
        
    

### Benefits of Classifier-Free Guidance

- **Flexibility**: It allows for more nuanced control over the generated output without the complexity of training a separate classifier.
- **Improved Quality**: By balancing between conditioned and unconditioned outputs, the generated images tend to be more relevant and of higher quality.
- **Diversity**: It can enhance the diversity of generated samples while still keeping them aligned with the intended themes or styles.

### Applications

- **Text-to-Image Generation**: Enhancing the relevance of images generated from textual descriptions.
- **Image Manipulation**: Allowing for creative control in modifying images based on user inputs or themes.

### Summary

Classifier-free guidance is a powerful technique that enables better control over the output of diffusion models, particularly in generating images based on text prompts. By leveraging both conditioned and unconditioned outputs, it enhances the quality, relevance, and diversity of generated samples.

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%203.png)

## DDPM(**Denoising Diffusion Probabilistic Models**)

In the above pseudo code you can see that DDPM(**Denoising Diffusion Probabilistic Models**) is a sampling algorithm this is where we are actually subtracting noise from the predicted image

Why extra noise is added during sampling 

## NN expects a noisy sample as input

This stabilizes the NN so it doesn’t collapse to something closer to the average dataset. 

### 1. **Exploration of the Output Space**:

- **Diversity**: Introducing noise during the sampling process helps the model explore a wider range of possible outputs. This is crucial for generating varied and interesting samples rather than producing the same result repeatedly.

### 2. **Maintaining Uncertainty**:

- **Stochastic Nature**: By adding noise, the model retains a level of uncertainty in its predictions. This is particularly important in generative tasks, where capturing the inherent variability in the data distribution is desired.

### 3. **Preventing Mode Collapse**:

- **Coverage of the Distribution**: Adding noise during sampling can help prevent mode collapse, a phenomenon where the model generates only a limited number of outputs. This is especially relevant in generative models like GANs and diffusion models.

### 4. **Smooth Transitions**:

- **Gradual Denoising**: In the context of diffusion models, noise is added to facilitate the transition from a noisy image back to a clean one. Each step in the reverse diffusion process involves gradually removing noise, allowing for smoother outputs.

### 5. **Encouraging Creative Outputs**:

- **Novelty**: In creative applications (like image generation), adding noise can lead to more novel outputs that deviate from typical training samples, fostering innovation and unexpected results.

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%204.png)

Architecture 

we use UNET architecture which actually works good for segmentation and similar task

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%205.png)

 

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%203.png)

1. Training Process

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%206.png)

Loss function Mean square error loss 

1. Controlling the model 

Controlling means adding context embedding which means adding text embedding in the model. classifier free training refers to the same concept 

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%207.png)

Faster sampling method

DDIM (**Denoising diffusion implicit models**)

You want more images fast

But sampling is slow because

- there are many timesteps
- each timesteps is dependent on the previous one(markovian)

Many new sampler address this problem of the speed 

one is called DDIM (denoising diffusion implicit model)

![image.png](https://github.com/sarveshamberkar/Diffusion_from_scratch/blob/main/assets/image%208.png)

DDIm is faster because it skips timesteps

it predicts a rough idea of the final output and then refines it with the denoising process
