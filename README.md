# kztk
Utilities, pre-trained models, and more.

## requirements
 - pytorch

## installation
> pip install --upgrade git+git://github.com/k15z/kztk.git

## examples
```
import kztk.toxic as toxic

result = toxic.classify("your post is stupid")
result == {
    'threat': 0.027044642716646194, 
    'insult': 0.2758590579032898,
    'obscene': 0.32889148592948914, 
    'hate': 0.04638497531414032, 
    'toxic': 0.698431670665741, 
    'severe': 0.0460701622068882
}
```
