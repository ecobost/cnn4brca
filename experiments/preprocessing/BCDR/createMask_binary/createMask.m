# Written by: Erick Cobos T (a01184587@itesm.mx)
# Based on: example_outlines.m
# Date: 22-Nov-2015

# Script to generate a mask (binary image with true/white where lesions are present
# and false/black elsewhere) for every image in the dataset, including those with 
# no lesions and those with more than one.

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

	% Create mask with all false
	filename = iminfo.image_filename{i};
	[M, N] = size(imread(filename));
	mask = zeros(M, N, 'logical');

	% For each lesion
	for j = 1: length(outlines.lesion_id)
		% If lesion in current mammogram.
		if(strcmp(filename, outlines.image_filename{j}))
		% Could also select for other lesion features such as malignancy.

			%Parsing the ouline of the lesion
			x = str2num(outlines.lw_x_points{j});
			y = str2num(outlines.lw_y_points{j});
 
			% Calculate the mask
			lesionMask = poly2mask(x, y, M, N);
			
			% Join it with the previous one
			mask = or(mask, lesionMask);
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
