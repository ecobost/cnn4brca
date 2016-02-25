# Written by: Erick Cobos T (a01184587@itesm.mx)
# Date: 25-Feb-2016
# Based on: Original createMask

# Script to generate a mask (grayscale image with 255 where lesions are present, 0
# for background and 127 for other breast tissue and 0 for background) for every
# image in the dataset, including those with no lesions or multiple lesions.

# Note: I use the poly2mask function from the 'image' package in Octave-Forge. You
# can use the provided .m or get the entire package (octave.sourceforge.net).

% Set some parameters
iminfoFilename = 'bcdr_d01_img.csv';
outlinesFilename = 'bcdr_d01_outlines.csv';
fileExt = '_mask.png'; # Will be added to the end of the filename.

% Loading the image info
f = fopen(iminfoFilename);
header = textread(iminfoFilename, '%s', 8, 'delimiter', ',');
lines = textscan(f, '%d %d %d %s %s %d %d %d', 'delimiter', ',', 'headerlines', 1);
iminfo = cell2struct(lines, header, 2); 
fclose(f);

% Loading the outlines info
f = fopen(outlinesFilename);
header = textread(outlinesFilename, '%s', 19, 'delimiter', ',');
lines = textscan(f, '%d %d %d %d %d %d %s %s %s %d %d %d %d %d %d %d %d %d %s', 'delimiter', ',', 'headerlines', 1);
outlines = cell2struct(lines, header, 2); 
fclose(f);

% For each image in the dataset
for i = 1 : length(iminfo.image_filename)

	% Load image
	filename = iminfo.image_filename{i};
	mammogram = imread(filename);

	% Create mask with all zeros
	[M, N] = size(mammogram);
	mask = zeros(M, N);

	% Signal breast area as 127
	mask(mammogram > 0) = 0.5;

	% For each lesion
	for j = 1: length(outlines.lesion_id)
		% If lesion in current mammogram.
		if(strcmp(filename, outlines.image_filename{j})) 
		% Could also select for other lesion features such as malignancy.

			%Parsing the ouline of the lesion
			x = str2num(outlines.lw_x_points{j});
			y = str2num(outlines.lw_y_points{j});
 
			% Calculate the lesion mask
			lesionMask = poly2mask(x, y, M, N);
			
			% Add it to the image mask
			mask(lesionMask) = 1;
		end
	end

	% Save mask
	imwrite(mask, [filename(1:end-4) fileExt]);

	% Report stats
	if(mod(i, 5) == 0)
		disp(sprintf("%d/%d: %s", i, length(iminfo.image_filename), filename));
		fflush(stdout);
	end
end

# Tests:
#	It rewrites the file if already written
#	It works fine when no lesion
# 	It works fine when more than one lesion
#	It works for a lot of data
