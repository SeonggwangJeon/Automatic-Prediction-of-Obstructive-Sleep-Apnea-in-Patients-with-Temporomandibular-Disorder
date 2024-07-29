import cv2
import numpy as np
import os

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []        
        return self.model(x)


class GradCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 device):

        self.target_layer = target_layer
        self.device = device

        if not (self.device is None):
            self.model = model.to(self.device)

        self.model = model.eval()
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)
        print(self.activations_and_grads)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self, grads):
        return np.mean(grads, axis=(2, 3))

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      activations,
                      grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None):

        if not (self.device is None):
            input_tensor = input_tensor.to(self.device)

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(activations, grads)

        cam = np.maximum(cam, 0) # relu

        result = []
        for img in cam:
            img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
        result = np.float32(result)
        return result

    def __call__(self,
                 input_tensor,
                 target_category=None):
        return self.forward(input_tensor,target_category)
    
def get_visuals(nd_img, grad_cam):
    ret = []
    #length = len(nd_img)
    length = len(grad_cam)
    for i in range(length):
        ret.append(show_cam_on_image(nd_img[i], grad_cam[i]))
    return ret

def explort_imgs(images, labels, predicts, parent_dir,section=0):
    idx = 0
    labels = labels.cpu().numpy()
    for image in images:
        file_path = os.path.join(parent_dir, f'{section}_label{labels[idx]}pred{predicts[idx]}.jpg')
        print(file_path)
        cv2.imwrite(file_path, image)
        idx += 1  
        section += 1
    print('exported..done')
    
def show_cam_threshold(up, down, image, cam):
    cam_down = cam < down
    cam_up = cam >= up
    
    cam_sum = np.multiply(cam_down,cam_up)
    plt.figure(figsize=(5,5))
    plt.imshow(show_cam_on_image(image, cam_sum))
    plt.show()