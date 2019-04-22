# Results from experimentation

- With Siamese Net structure:


  1. If the branch is a **simple convolutional network** with `75` epochs:

<center>

| loss type         	| train accuracy 	| test accuracy 	|
|-------------------	|----------------	|---------------	|
| auxiliary loss    	| 92.4%          	| 85.8%         	|
| no auxiliary loss 	| 99 %           	| 82%           	|

</center>
  2. If the branch is a **residual network** with `75` epochs:

<center>

| loss type         	| train accuracy 	| test accuracy 	|
|-------------------	|----------------	|---------------	|
| auxiliary loss    	| 100%          	| 89.8%         	|
| no auxiliary loss 	| 100 %           | 82.5%           	|


</center> Maybe test with the structure that Fanny uses
