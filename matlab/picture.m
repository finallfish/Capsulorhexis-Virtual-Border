% DSQ
% I = imread('0001.jpg');
% I1 = rgb2gray(I);
% I2 = I1(130:155,262:287);
% I2 = 255 - I2;
% mesh(I2)
% 
% I3 = I1(115:140 ,422:447);
% I3 = 255-I3;
% mesh(I3)
% 
% I4 = I1(314:339 ,268:293);
% I4 = 255-I4;
% mesh(I4)


% FAM 
I = imread('0001.jpg');
I1 = rgb2gray(I);
I2 = I1(95:125,253:283);
imwrite(I2,'zuoshang.jpg');
I2 = 255 - I2;
mesh(I2)

I3 = I1(352:382 ,291:321);
imwrite(I3,'xia.jpg');
I3 = 255-I3;
mesh(I3)

I4 = I1(108:138 ,460:490);
imwrite(I4,'youshang.jpg');
I4 = 255-I4;
mesh(I4)




% imwrite(I2,'zuoshang.jpg');
% imwrite(I3,'youshang.jpg');
% imwrite(I4,'zuoxia.jpg');