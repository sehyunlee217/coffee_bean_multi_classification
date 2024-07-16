## Classifying coffee beans based on level of roasting using CNN Models (Effnet)  
### [Live Preview on Hugging Face](https://huggingface.co/spaces/sehyunlee217/Coffee_Bean_Classifier)
![Screenshot 2024-07-15 at 17 20 24](https://github.com/user-attachments/assets/8736f89c-9168-476e-8bd6-a0d2961f6e5f)
___
Drinking coffee in the morning is one of the most important things that many do to start the day. 

While several elements contribute to a good cup of coffee, understanding the types of beans and their roasting levels is key to mastering your morning coffee.

For this dataset, four roasting levels with its bean type are addressed: Green/Unroasted and lightly roasted coffee beans are Laos Typica Bolaven. Doi Chaang are the medium roasted, and Brazil Cerrado are dark roasted. All coffee beans are Arabica beans.

### Green/Unroasted 
![green](https://github.com/user-attachments/assets/24655e7f-5ce2-4133-bc9e-7327f1c115c0)

### Lightly Roasted
![light](https://github.com/user-attachments/assets/495137d1-9d33-4276-a455-3457fdf73067)

### Medium Roasted
![medium](https://github.com/user-attachments/assets/392f7f6d-074c-47a9-a3ac-64f1779224d6)

### Dark Roasted
![dark](https://github.com/user-attachments/assets/0212ade6-37a4-4633-9072-0424ffcf15dc)

Two CNN models were tested against each other to determine which model was more suitable for this multi-classification problem. As the problem involves classifying four different classes and the model was to be run on Hugging Face Spaces, a smaller model was deemed adequate and preferred due to size limitations. Effnet B1 and Effnet V2 S were evaluated based on their loss and accuracy.

Both models were trained on 300 images per class (4 roasting types) and tested on 100 images per class.

Since this is a multi-classification problem, the cross-entropy loss function was used with the Adam optimizer. Both models were trained for 5 and 10 epochs, which was sufficient for this straightforward classification problem.

### Test Accuracy (Effnet B1 v.s Effnet_v2_s)
![Screenshot 2024-07-15 at 17 33 16](https://github.com/user-attachments/assets/cfee5a0c-4d86-42c3-b525-c64240242263)

### Test Loss (Effnet B1 v.s Effnet_v2_s)
![Screenshot 2024-07-15 at 17 33 30](https://github.com/user-attachments/assets/dc3714b4-b4e0-4692-9081-2d12ba2f7d83)

Both models performed very well, achieving over 98% test accuracy and demonstrating very low loss. 
However, given that Effnet B1 has a size of 26MB compared to Effnet V2 S’s 81MB, Effnet B1 was deemed more appropriate for this project.

The prediction probabilities were not the highest for medium vs. dark roasted beans, as both are often easily mistaken for one another. 
While the overall prediction labels were accurate, the model lacked confidence when differentiating between these two types. This difficulty was understandable upon examining the photos.

#### One is dark and one is medium here, but it is difficult even for the human eye to distinguish between the two due to shadows and image quality.
![medium (20)](https://github.com/user-attachments/assets/e124eb0f-b7ea-43d8-8c8d-a0b88606cc4f)
![dark (64)](https://github.com/user-attachments/assets/5c998f89-ccef-4975-8d7d-eeedbecf431a)

In conclusion, while both CNN models demonstrated high accuracy and low loss, Effnet B1 was selected as the more suitable model due to its smaller size, making it more practical for deployment on Hugging Face Spaces. 
Despite the overall high prediction accuracy, distinguishing between medium and dark roasted beans proved to be challenging, due to shadows in the images. 
Future improvements could focus on enhancing the model’s ability to differentiate between these closely related classes and also adding different types of beans and its shapes as well. 

Dataset from [Coffee Bean Data](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) | Ontoum, S., Khemanantakul, T., Sroison, P., Triyason, T., & Watanapa, B. (2022). Coffee Roast Intelligence. arXiv preprint arXiv:2206.01841.
