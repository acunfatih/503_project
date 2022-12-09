%colorsOrdered

%Taken from: https://sashamaps.net/docs/resources/20-colors/
function [value,color,colorList] = ColorsOrdered(index)
    index = floor(index);
    if index > 21
        index = mod(index-1,21)+1;
    elseif index < 1
        index = 1;
    end
        
    Yellow = [255, 225, 25];
    Blue = [0, 130, 200];
    Orange = [245, 130, 48];
    Lavender = [220, 190, 255];
    Maroon = [128, 0, 0];
    Navy = [0, 0 ,128];
    Red = [230, 25, 75];
    Green = [60, 180, 75];
    Cyan = [70, 240, 240];
    Magenta = [240, 50, 230];
    Pink = [250, 190, 212];
    Teal = [0, 128, 128];
    Brown = [170, 110, 40];
    Beige = [255, 250, 200];
    Mint = [170, 255, 195];
    Purple = [145, 30, 180];
    Lime = [210, 245, 60];
    Olive = [128, 128, 0];
    Apricot = [255, 215, 180];
    Grey = [128, 128, 128];
    Black = [0,0,0];
    
    colorVals = [Yellow;Blue;Orange;Lavender;Maroon;Navy;Red;Green;Cyan;Magenta;Pink;Teal;Brown;Beige;Mint;Purple;...
        Lime;Olive;Apricot;Grey;Black];

    colorList = {'Yellow','Blue','Orange','Lavender','Maroon','Navy','Red','Green','Cyan','Magenta','Pink','Teal',...
        'Brown','Beige','Mint','Purple','Lime','Olive','Apricot','Grey','Black'};
    
    color = colorList{index};
    value = colorVals(index,:)./255;
end
