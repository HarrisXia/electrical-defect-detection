function wire_defect(img)

[h,w,c]=size(img);
im_h = h*0.5;im_w = w*0.5;
p_h = im_h*0.08;p_w = im_w*0.08;
img= imresize(img,[im_h,im_w],'bicubic');
threold_min=100;
threold_max=200;
im_gray = double(rgb2gray(img));
    %im_gray = histeq(im_gray1);
    %im_gray = adapthisteq(im_gray1);
    im = DUCO_RemoveBackGround(im_gray,15,0);
% 
%     figure(1)
%     subplot(2,2,1);imshow(img);title('ori image');
%     subplot(2,2,2);imshow(im_gray,[]);title('gray image')
%     subplot(2,2,3);imshow(im,[]);title('remove light')
%     
    
    im_mean = mean(im(:));
    %imhist(im(:,:,1)); 
    img_th1=roicolor(im,threold_min,threold_max); 
    
     th=graythresh(im/255);
     img_th2=im2bw(im/255,th);
     
    img_th3=edge(im,'Canny',0.3);
    
%     se = strel('square',2);
%     img_th4 = imdilate(im,se)-imerode(im,se);

 
img_th1 = afterprocessing(img_th1);
img_th2 = afterprocessing(img_th2);
img_th3 = afterprocessing(img_th3);

 img_avg=(img_th1+img_th2+img_th3)/3;
 th_avg=graythresh(img_avg);
  img_avg=im2bw(img_avg,th_avg);
  
% figure(2)
% subplot(2,2,1);imshow(img_avg);title('avg result');
% subplot(2,2,2);imshow(img_th1);title('hand threold')
% subplot(2,2,3);imshow(img_th2);title('OSTU')
% subplot(2,2,4);imshow(img_th3);title('canny edge')
%% line detect 
[H, theta, rho]= hough(img_avg); 
peak=houghpeaks(H,4); 
lines=houghlines(img_avg,theta,rho,peak,'FillGap', 30, 'MinLength', 400); 
figure;imshow(img),title('Hough Transform Detect Result'),hold on    
start_x= [];start_y =[];end_x = [];end_y=[];
%% ȱ�ݼ��
i=1;
for i=1:2
    xy=[lines(i).point1;lines(i).point2];   
    point1=lines(i).point1;
    point2=lines(i).point2;
    plot(xy(:,1),xy(:,2),'blue','LineWidth',2); 
% end
    start_x(1) = point1(1);
    if point1(2)-p_h/2>0
        start_y(1)  =point1(2)-p_h/2;
    else
        start_y(1)  =point1(2)
    end
    end_x(1) = point2(1);
    end_y(1) = point2(2)-p_h/2;
    k=(end_y(1)-start_y(1))/(end_x(1)-start_x(1));

    j=1;
    proposal1=[];
    proposal2=[];
    output=[];
    Areas=[];
while(1)  
    if  (point1(2)-p_h/2+p_h)>=im_h 
        if start_x(j)<0 ||start_y(j)<0 || start_y(j)-p_h<0
            break
        end 
       proposal1{j}=Y((start_y(j)-p_h:start_y(j)),start_x(j):(start_x(j)+p_w));%�Ҷ�ͼ����
       proposal2{j}=img_avg((start_y(j)-p_h:start_y(j)),start_x(j):(start_x(j)+p_w));%��ֵͼ�����
     ;
       output{j}= Imentropy(proposal1{j});
       [L1,num1] = bwlabel(proposal2{j}, 8);
        for h=1:num1
            Areas(j) = sum(L1(:)==h); % ���
        end
        j=j+1;
        start_x(j)= start_x(j-1)+p_w;  %x����
        start_y(j) = start_y(j-1)+floor(k*p_w); %y����
        if start_y(j)<0
            break
        end         
    end 
    
    if (start_x(j)+p_w)>=im_w || (start_y(j)+p_h)>=im_h 
        break
    end 
    if start_x(j)<0 ||start_y(j)<0
        break
    end 
    
    if (point1(2)-p_h/2+p_h)<im_h 
       proposal1{j}=im_gray(start_y(j):(start_y(j)+p_h),start_x(j):(start_x(j)+p_w));%�Ҷ�ͼ����
       proposal2{j}=img_avg(start_y(j):(start_y(j)+p_h),start_x(j):(start_x(j)+p_w));%��ֵͼ�����
    
       output{j}= Imentropy(proposal1{j});
       [L1,num1] = bwlabel(proposal2{j}, 8);
       for h=1:num1
           Areas(j) = sum(L1(:)==h); % ���
       end
        j=j+1;
        start_x(j)= start_x(j-1)+p_w;  %x����
        start_y(j) = start_y(j-1)+floor(k*p_w); %y����
    end

end  

 out=cell2mat(output);
 th1=mean(out)
 th2=mean(Areas)
 j=1;
 
 for j=1:length(Areas)
         if  Areas(j)>th2*1.02|| Areas(j)<th2/1.5
             if output{j}>th1*1.02 || output{j}<th1/1.5
                rectangle('Position',[start_x(j),start_y(j),p_w,p_h],'LineWidth',2,'LineStyle','-','EdgeColor','r');
             end
         end
 end

 
 
 
 end



