
#### Installation:
Anaconda Python Environment <br/>
version is working for CPU or [GPU] <br/>
Python 3.8 <br/>
torchvision 0.16.1 (pip3 install torchvision==0.16.1) <br/>
torch 2.1.1 or [torch 2.1.1+cu121] (https://pytorch.org/get-started/locally/) <br/>

# Ball-Shear-ANN-Regression
## Prototype Ballshear ANN Regression on ASM Wirebonder
## Resource Link :<br/>
Random Forest regression model : <br/>
https://levelup.gitconnected.com/random-forest-regression-209c0f354c84
ANN regression model :<br/> 
https://colab.research.google.com/drive/1eje9zILprgVmohMN7cKykI3fn4FBRPnF <br/>
Onehot encoder :<br/>
https://towardsdatascience.com/columntransformer-in-scikit-for-labelencoding-and-onehotencoding-in-machine-learning-c6255952731b <br/>
https://stackoverflow.com/questions/58087238/valueerror-setting-an-array-element-with-a-sequence-when-using-onehotencoder-on

## ANN Evaluation <br/>
In-HiddenLayer-Out : BS : EP = Loss% <br/>
35-100-35-1 : 32 : 200 = 12.278% <br/>
35-100-100-35-1 : 32 : 200 = 12.285% <br/>
35-140-100-35-1 : 32 : 200 = 12.259% <br/>
35-175-100-35-1 : 32 : 200 = 12.197% <br/>
35-210-35-1 : 32 : 200 = 12.254% <br/>


