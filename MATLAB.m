start=30000;     %从数据集中从start开始取
stop=32000;      %到stop结束

bia=randi([1,1],stop-start+1,1);
x=input_h(start:stop,:)*10000000;

output=x*weight_1+bia*biase_1;%第一层
[m, n] = size(output);  % 记录矩阵大小//relu
AA = output(:);   % 矩阵拉直成一维向量
a = find(AA<0);   % 找出所有小于0的数
AA(a) = 0;   % 将小于0的数用0替代
output = reshape(AA,m,n);  % 恢复矩阵形式

output = output*weight_2+bia*biase_2;%第二层
[m, n] = size(output);  % 记录矩阵大小//relu
BB = output(:);   % 矩阵拉直成一维向量
b = find(BB<0);   % 找出所有小于0的数
BB(b) = 0;   % 将小于0的数用0替代
output = reshape(BB,m,n);  % 恢复矩阵形式

output = output*weight_3+bia*biase_3;%输出层
[m, n] = size(output);  % 记录矩阵大小//相当于后续处理的集合
CC = output(:);   % 矩阵拉直成一维向量
c = find(CC<0);   % 找出所有小于0的数
d = find(CC>0);   % 找出所有大于0的数
CC(c) = 0;   % 将小于0的数用0替代
CC(d) = 1;   % 将大于0的数用1替代
output = reshape(CC,m,n);  % 恢复矩阵形式

y=output_mode(start:stop,:);
loss=y-output;
[m, n] = size(loss);  % 记录矩阵大小
EE = loss(:);   % 矩阵拉直成一维向量
f = find(EE==0);   % 找出所有等于0的数
accuracy=length(f)/length(EE);
loss = reshape(EE,m,n);  % 恢复矩阵形式
