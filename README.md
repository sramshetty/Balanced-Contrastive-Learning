# Supervised Contrastive Learning for Long-tail Learning

Pytorch implementation of our 7643 project. 

## Abstract
Deep learning has achieved remarkable success when trained on a balanced dataset. However, real-world datasets are often imbalanced, i.e., few classes have many samples and a great number of classes have few samples. Training on such imbalanced datasets normally leads to poor generalization since the model is biased toward head classes with massive training data. In this paper, we improve the imbalanced dataset recognition accuracy via supervised contrastive learning (SCL) loss since SCL outperforms its cross-entropy counterparts in recent studies. We first study the balanced contrastive learning loss that aims to mitigate the accuracy drop when directly applying SCL into long-tail study. Next, we thoroughly research the impact of different components in our pipeline, e.g., data augmentation, network backbones, and cost-sensitive loss functions. Overall, we provide a systematic study on adopting SCL to mitigate the long-tail learning issue and improve recognition performances.

### CIFAR10
python main_anthony.py --dataset "cifar10"  --arch "resnet50" --lr 0.1 --epochs 100 --wd 5e-4 --cos True --num_classes 10

### CIFAR-100
python main_anthony.py --dataset "cifar100"  --arch "resnet50" --lr 0.1 --epochs 100 --wd 5e-4 --cos True --num_classes 100