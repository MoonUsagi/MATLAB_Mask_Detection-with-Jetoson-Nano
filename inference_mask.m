function inference_mask()
%#codegen

persistent mynet

%Create a jetson object to access the board and its peripherals
hwobj = jetson;
w = webcam(hwobj,1,[640 480]);
d = imageDisplay(hwobj);

%Load the trained network from a MAT-file
if isempty(mynet)
    mynet = coder.loadDeepLearningNetwork('COVID19_Mask_yolo.mat');
end
text_str = char("      Great! Be safe!       ");
text2_str = char("      Wear the MASK!      ");
fps = 0;
avgfps =0;
resz =[224 224];

for i = 1:1e5
    img = snapshot(w);
    sz = size(img);
    img2 = imresize(img, resz);
    %
    tic;
    [bboxes,scores,labels] = detect(mynet,img2,'Threshold',0.55);
    newt = toc;
    
    fps = .9*fps + .1*(1/newt);
    avgfps = [avgfps, fps];
    
    [~,idx] = max(scores);
    % Annotate detections in the image.
    
    if ~isempty(bboxes)
        bboxf = bboxre(bboxes, sz, resz);
        
            outImg = insertObjectAnnotation(img,'Rectangle',bboxf(idx,:),labels{idx});
            outImg = insertText(outImg, [200, 1],text_str, 'FontSize', 25,'TextColor',[1 1 1],'BoxColor', [100 255 100]);
            else
        outImg = img;
        outImg = insertText(outImg, [200, 1],text2_str, 'FontSize', 25,'TextColor',[1 1 1],'BoxColor', [255 50 50]);
    end
 
    outImg = insertText(outImg, [1, 1],sprintf('FPS: %2.2f', mean(avgfps)), 'FontSize', 10, 'BoxColor', [255 255 0], "BoxOpacity", 0.3);  
    image(d,outImg);
end
    function bbox = bboxre(bbox, sz, targetSize)
        bbox(:,1) = bbox(:,1)*sz(2)/targetSize(2);
        bbox(:,2) = bbox(:,2)*sz(1)/targetSize(1);
        bbox(:,3) = bbox(:,3)*sz(2)/targetSize(2);
        bbox(:,4) = bbox(:,4)*sz(1)/targetSize(1);
    end
end

