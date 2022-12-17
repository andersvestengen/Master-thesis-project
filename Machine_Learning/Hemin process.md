Loading from files is done by:

#This will load the image from file
def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

#Apply this to correctly convert to tensor
def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


getitem method(idx):
    image = load_image(im_path[idx])
    mask  = load_image(im_path[idx])

    return to_float_tensor(image), to_float_tensor(mask)



#Using the image during training is done like this:

Don't really need to change anything here.



#Turning the images back, to display them:

image_out = tensor_image.mul(255).add_(0.5).clamp_(0,255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
im = Image.fromarray(image_out)
im.save(path)

#Add this later if no issues.
just use scaled_image = np.around((image * 255), 0) 

to get the original image back.