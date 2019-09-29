load('prob_map.mat');
fid=fopen('101data.txt','w');
[b1 b2]=size(prob_map);
 
 
for i=1:b1
    for j=1:b2
       fprintf(fid,'%10f',prob_map(i,j));
    end 
    fprintf(fid,'\n');
end
fclose(fid);