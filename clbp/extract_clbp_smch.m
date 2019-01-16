function  CLBP_SMCH = test_one_image(filename)

img = load(filename);
Gray = img.Gray;

% Radius and Neighborhood
R=1;
P=8;

% generate CLBP features
patternMappingriu2 = getmapping(P,'riu2');

[CLBP_S,CLBP_M,CLBP_C] = clbp(Gray,R,P,patternMappingriu2,'x');
    
% Generate histogram of CLBP_S/M/C
CLBP_MCSum = CLBP_M;
idx = find(CLBP_C);
CLBP_MCSum(idx) = CLBP_MCSum(idx)+patternMappingriu2.num;
CLBP_SMC = [CLBP_S(:),CLBP_MCSum(:)];
Hist3D = hist3(CLBP_SMC,[patternMappingriu2.num,patternMappingriu2.num*2]);
CLBP_SMCH = reshape(Hist3D,1,numel(Hist3D));
