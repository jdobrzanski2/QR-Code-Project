% ELEX 7815 QR code project
% Author: Joseph Dobrzanski
% Version: 2019-11-15

% Project based on https://web.stanford.edu/class/ee368/Project_06/project.html
%	Training images 1 - 12 are from the link above, images 13 - 18 were added by
%	Joseph to test the program. Loop that iterates through test images modified 
%	from evaluate.m in above link

% first part of program to scan through "training_#.jpg" images 
% to show images of each QR code, set a breakpoint at end of loop and set
%   "SHOW_IM" to "1"

NUM_IMAGES = 18; 	% 18 training images to iterate through
SHOW_IM = 1; 		% 1 = show images in pop-up windows, 0 = do not show images (for debugging/demonstration purposes)
IMAGE_NAME = 'training';

image_run_time = zeros(1,NUM_IMAGES); % execution time of your detection routine for each image

% for testing one image
%{
img = imread('test12.jpg');
[data_detected,origin_detected] = QR_scanner(img, img_num, SHOW_IM);
%}

% for iterating through all test images. Program assumes that image is
% named "training_X.jpg" where "X" is the number of the test image
%   uses some parts of the Stanford "evaluation.m" to do this
for img_index = 1:NUM_IMAGES
   
    % read the image and the ground truth data
    img = imread(sprintf('%s_%d.jpg', IMAGE_NAME, img_index));   

    % execute your detection routine
    start_time = clock;

    [data_detected,origin_detected] = QR_scanner(img, img_index, SHOW_IM);
    
    image_run_time(img_index) = etime(clock, start_time);
end
total_run_time = sum(image_run_time);
fprintf('%s %.3d','Scan complete. Total execution time = ', total_run_time);


% QR_scanner - detect if QR codes are in the inputted image
% @param img_QR_indexread - image to find QR codes in
% @param image_num - which image number this is (naming purposes)
% @param show_images - debug variable for if you want image w QR code highlighted to appear in window
% @return data_out - QR code data extracted
% @return coord_out - coordinates the QR code(s) are located in the image
function [data_out, coord_out] = QR_scanner(img_QR_indexread, image_num, show_images)
	% close any other windows
	close all;
	
	% resize image to 480x??? (make number of rows 480; columns of image is scaled accordingly)
	des_row = 480;
	[check_row,~] = size(img_QR_indexread);
	if (check_row ~= des_row)
		scale_factor = des_row/check_row;
		img_QR_index_original = imresize(img_QR_indexread, scale_factor);
		img_QR_indexgy = rgb2gray(img_QR_index_original);
	else
		img_QR_indexgy = rgb2gray(img_QR_indexread);
		img_QR_index_original = img_QR_indexread;
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% THRESHOLDING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% prepare image so potential QR code blobs pop out %%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% parameters for filtering out blobs outside a given size range
	min_px_size = 30;
	max_px_size = inf;

	% boost edges
	img_QR_indexboost = booster(img_QR_index_original);
	img_QR_indexgy_boost = booster(img_QR_indexgy);

	% threshold image
	img_QR_indexth = rgb_thresh(img_QR_indexboost);
	img_QR_indexgy_th = gray_thresh(img_QR_indexgy_boost);

	% remove blobs touching border of image
	img_QR_indexadj = imadjust(imclearborder(img_QR_indexth));
	img_QR_indexgy_adj = imadjust(imclearborder(img_QR_indexgy_th));

	% fill holes of remaining blobs (be careful with is one in case QR code
	% completely surrounded by blob that isn't removed prior to this step
	img_QR_indexfill = img_QR_indexadj;
	img_QR_indexgy_fill = img_QR_indexgy_adj;

	% Increase rectangular size (helps with cases such as 'training_10')
	img_QR_indexfill = imdilate(img_QR_indexfill, strel('square',2));

	% Remove blobs based on a minimum and maximum size
	mask_th = bwareafilt(logical(img_QR_indexfill), [min_px_size, max_px_size]);
	mask_gy_th = bwareafilt(logical(img_QR_indexgy_fill), [min_px_size, max_px_size]);

	% create AND'ed image from both thresholds for improved results
	mask_comb_AND = mask_th & mask_gy_th;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% ISOLATING QR CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Evaluate blobs in thresholded image to see if it's a QR code %%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% set-up variables for storing potential QR code in sub-images
	QR_index = 1;
	clear QR_codes;
	clear QR_codes_rect;
	[im_row_size, im_col_size] = size(mask_comb_AND);

	% variable for isolating blobs based on major_axis-to-minor_axis ratio
	ax_ratio_min = 4;
	ax_ratio_max = 10;

	% remove blobs based on length-width ratio
	[labeledImage, numberOfBlobs] = bwlabel(mask_comb_AND, 8); % needed for blob removal
	length_stats = regionprops(logical(mask_comb_AND),'MajorAxisLength','MinorAxisLength');
	masked = mask_comb_AND;
	axis_ratio = [length_stats.MajorAxisLength]./[length_stats.MinorAxisLength];
	
	for k = 1:numberOfBlobs
		% isolate which blobs are shown based on if ratio a certain size
		if ((axis_ratio(k) < ax_ratio_min) || (axis_ratio(k) > ax_ratio_max))
			% remove disqualified blobs from image
			for row_ = 1:im_row_size %go through each row of image
				for col_ = 1:im_col_size % go through each col of image 
					if (labeledImage(row_,col_) == k)
						masked(row_,col_) = 0;
					end
				end
			end  
		end
	end

	% get appropriate information out of remaining blobs
	second_check_stats = regionprops(logical(masked),'Centroid', 'Orientation', 'MajorAxisLength', 'Area');
	blob_centroid = reshape([second_check_stats.Centroid], 2,[]).'; % get into correct matrix shape
	blob_orientation = [second_check_stats.Orientation];
	blob_majoraxis = [second_check_stats.MajorAxisLength];
	blob_area = [second_check_stats.Area];

	dist_coeff = 0.75; % for filtering based on distance

	% for filtering based on angle
	angle_tol = 27;
	min_angle = 90 - angle_tol;
	max_angle = 90 + angle_tol;
	% for filtering based on similarity in area
	area_tol = 1.7;
	% for choosing a subsection of an image (centred around each QR code)
	select_area_coef = 2;

	% increase QR size by a given multiplier to make further processing better
	min_QR_row_size = 125; % min number of rows sub-image should have
	scaler = 1.5;

	% isolate rectangles common to QR codes
	masked = imfill(masked, 'holes');
	[labeledImage, ~] = bwlabel(masked, 8); % needed for blob removal
	for i = 1:length(blob_orientation)
		for j = i:length(blob_orientation)        
			% see if two blobs are approximately orthogonal to each other
			diff_angle = abs(blob_orientation(i) - blob_orientation(j));
			if ~((diff_angle > min_angle) && (diff_angle < max_angle))
				continue; % disqualify blob pairs that are not
			end
			
			% see if two blobs within a certain distance of each other
			row_dist = blob_centroid(i,1) - blob_centroid(j,1);
			col_dist = blob_centroid(i,2) - blob_centroid(j,2);
			dist = sqrt(row_dist ^ 2 + col_dist ^ 2);
			larger_length = max([blob_majoraxis(i), blob_majoraxis(j)]);
			if ~(dist < dist_coeff * larger_length)
				continue;
			end
				
			% see if two blobs have a similar area if area of larger area_tol+times larger, disqualify
			s_area = min([blob_area(i) blob_area(j)]);
			b_area = max([blob_area(i) blob_area(j)]);
			if (b_area / s_area >= area_tol)
				continue;
			end
			
			% create selection area
			l_row = round(min([blob_centroid(i,2), blob_centroid(j,2)]) - select_area_coef * larger_length);
			u_row = round(max([blob_centroid(i,2), blob_centroid(j,2)]) + select_area_coef * larger_length);
			l_col = round(min([blob_centroid(i,1), blob_centroid(j,1)]) - select_area_coef * larger_length);
			u_col = round(max([blob_centroid(i,1), blob_centroid(j,1)]) + select_area_coef * larger_length);
			% limit selection area if out of bounds
			if (l_row < 1); l_row = 1; end
			if (l_col < 1); l_col = 1; end
			if (u_row > im_row_size); u_row = im_row_size; end
			if (u_col > im_col_size); u_col = im_col_size; end

			% create sub-images
			sub_I = logical(masked(l_row:u_row, l_col:u_col));
			QR_I = img_QR_indexgy(l_row:u_row, l_col:u_col); 
			sub_labeledImage = labeledImage(l_row:u_row, l_col:u_col);
			[sub_row,sub_col] = size(sub_I);

			% remove unwanted blobs from sub-image
			for row_blob = 1:sub_row %go through each row of sub-image
				for col_blob = 1:sub_col % go through each col of sub-image 
					if (sub_labeledImage(row_blob, col_blob) ~= i) && (sub_labeledImage(row_blob, col_blob) ~= j)
						sub_I(row_blob, col_blob) = 0;
					end
				end
			end  

			% rotate sub-images
			sub_I = imrotate(sub_I, -blob_orientation(i));
			QR_I = imrotate(QR_I, -blob_orientation(i));

			% do shear correction on sub-images
			shear_deg = 90-(blob_orientation(i)-blob_orientation(j));
			shear_val = tan(deg2rad(shear_deg));
			tform_shear = affine2d([1 0 0; -shear_val 1 0; 0 0 1]);
			sub_I = imwarp(sub_I,tform_shear);
			QR_I = imwarp(QR_I,tform_shear, 'FillValues', 255);

			% get states needed for next operations
			sub_stat = regionprops(sub_I,'Centroid');
			sub_centroid = reshape([sub_stat.Centroid], 2,[]).';
			[sub_row,sub_col] = size(sub_I);

			% using upper centroid of rect, find # of pixel transitions
			top_row = round(min([sub_centroid(1,2), sub_centroid(2,2)]));  
			bot_row= round(max([sub_centroid(1,2), sub_centroid(2,2)]));  
			last_px = sub_I(top_row,1);
			last_px_check = sub_I(bot_row,1);
			transition = 0;
			transition_check = 0; % to ensure no mirror rect pairs included

			% see where centroids are w.r.t. each other
			% whether result is +ve or -ve ro row/col shows orientation
			row_dist = sub_centroid(1,1)-sub_centroid(2,1);
			col_dist = sub_centroid(1,2)-sub_centroid(2,2);
			%{
				1   2   3   4
			  1     X         
			  2         X    
			  3     O          
			  4 O              
			 X = 90 deg or -90 deg
			 O =  0 deg or 180 deg
			%}

			% X scenario where 90 degree or -90 degree out of rotation
			if (((row_dist > 0) && (col_dist > 0)) || ((row_dist < 0) && (col_dist < 0)))
				% scan through all cols in one row to see how many
				% pixel transitions threre are
				for index = 1:sub_col
					if (sub_I(top_row,index) ~= last_px)
						last_px = sub_I(top_row,index);
						transition = transition+1;
					end
					if (sub_I(bot_row,index) ~= last_px_check)
						last_px_check = sub_I(bot_row,index);
						transition_check = transition_check+1;
					end
				end

				% based on # transitions, do certain rotation
				if (transition == 2) && (transition_check >= 4)
					% |       -->        |
					% |--    90 CCW     --
					req_angle = -90; % blob_orientation(i)+180; 
					%disp('1');
				elseif (transition >= 4) && (transition_check == 2)% includes greater than 4 in case a rect tilted too much
					% --|    -->      |
					%   |   90 CW    --
					req_angle = 90;% 180+blob_orientation(j); 
					%disp('2');
				else
					disp('E_1');%ERROR, wierd amount of transitions found
					continue;
				end

			% O scenario where 180 degree or 0 degree out of rotation
			elseif (((row_dist < 0) && (col_dist > 0)) || ((row_dist > 0) && (col_dist < 0)))
				right_col = round(max([sub_centroid(1, 1), sub_centroid(2, 1)]));
				left_col = round(min([sub_centroid(1, 1), sub_centroid(2, 1)]));
				% scan through all rows in one col to determine how
				% many pixel transitions there are
				for index = 1:sub_row
					if (sub_I(index,right_col) ~= last_px)
						last_px = sub_I(index,right_col);
						transition = transition + 1;
					end
					if (sub_I(index,left_col) ~= last_px_check)
						last_px_check = sub_I(index,left_col);
						transition_check = transition_check + 1;
					end
				end

				% based on # transitions, do certain rotation
				if (transition == 2) && (transition_check >= 4)
					% __      -->       |
					% |     180 deg    --
					req_angle = 180;%(blob_orientation(i)+180); 
					%disp('3');
				elseif (transition >= 4) && (transition_check == 2)% includes greater than 4 in case a rect tilted too much
					%  |     -->       |
					% --    0 deg     --
					req_angle = 0; %blob_orientation(i);                         
					%disp('4');
				else
					disp('E_2'); %ERROR, wierd amount of transitions found
					continue;
				end   
			else
				disp('E_3');%ERROR, wierd orientation of centroids found
			end

			% rotate image again by required amount to orient correctly
			sub_I = imrotate(sub_I, -req_angle);
			QR_I = imrotate(QR_I, -req_angle);      

			% get info on rotated sub-images needed for trimming
			sub_stat = regionprops(sub_I,'Centroid');
			sub_centroid = reshape([sub_stat.Centroid], 2, []).';
			[sub_img_QR_indexrow, sub_img_QR_indexcol] = size(sub_I);

			% trim area to better include only QR code
			l_row = round(min([sub_centroid(1, 2), sub_centroid(2, 2)]) - 1 * larger_length); 	%1
			u_row = round(max([sub_centroid(1, 2), sub_centroid(2, 2)]) +.4 * larger_length); 	%.4
			l_col = round(min([sub_centroid(1, 1), sub_centroid(2, 1)]) - 1.6 * larger_length); % 1.6
			u_col = round(max([sub_centroid(1, 1), sub_centroid(2, 1)]) + .4 * larger_length); 	% 0.4
			if (l_row < 1); l_row = 1; end
			if (l_col < 1); l_col = 1; end
			if (u_row > sub_img_QR_indexrow); u_row = sub_img_QR_indexrow; end
			if (u_col > sub_img_QR_indexcol); u_col = sub_img_QR_indexcol; end   
			sub_I = sub_I(l_row:u_row, l_col:u_col);
			QR_img_QR_indexcrop = QR_I(l_row:u_row, l_col:u_col);

			% if size of QR code small, make larger to help ID the QR segments
			if (u_row-l_row < min_QR_row_size)
				sub_I=imresize(sub_I, scaler);
				QR_img_QR_indexcrop=imresize(QR_img_QR_indexcrop, scaler);
			end

			% store QR codes in a variable
			QR_coords{1,QR_index} = [blob_centroid(i,1) blob_centroid(i,2)];
			QR_codes{1,QR_index} = QR_img_QR_indexcrop;
			QR_codes_rect{1,QR_index} = sub_I;
			QR_index = QR_index+1;
		end    
	end

	% if no QR codes found, exit from function
	if (QR_index == 1)
		disp('no QR codes found');
		return;
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% INTERPRET QR CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% From the QR codes detected, pull out encoded value %%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	th_value = 50; % for a simple threshold

	for img_QR_index = 1:size(QR_codes, 2)
		% create local copies from QR_codes list
		QR_local = QR_codes{img_QR_index};
		QR_rect_local = QR_codes_rect{img_QR_index};

		% boost edges, then threshold image
		QR_local_th = booster(QR_local);
		QR_local_th = th(QR_local_th,QR_rect_local);

		% ensure middle of large blobs filled
		QR_local_bk = logical(simple_th(QR_local, th_value));
		QR_local_th = QR_local_th|QR_local_bk;

		% clean up QR sub-image further
		QR_cleaned_adj = cleanup(QR_local_th);
		QR_rect_local = cleanup(QR_rect_local);
		
		% find corners of QR code by using centroids of mask 
			% 1st rect in length_stats is horizontal rect ; h rect is 5px in QR code
			% 2nd rect in length_stats is vertical rect ; v rect is 7px in QR code
		h_r = 1; h_r_len = 5;
		v_r = 2; v_r_len = 7;
		mask_stat = regionprops(QR_rect_local,'Centroid');
		QR_stat = regionprops(logical(QR_cleaned_adj),'Centroid');
		mask_centroid = reshape([mask_stat.Centroid], 2,[]).';
		QR_centroid = reshape([QR_stat.Centroid], 2,[]).';
		[mask_row, ~] = size(QR_rect_local);

		% assume thresholded area a bit larger than it should be
		px_border_corr = 1;

		% determine height of vertical rectangle (start counting at first detected
		% white px, stop counting at next detected black px)
		cut = 0;
		count = 0;
		for row = 1:mask_row
			if QR_rect_local(row,round(mask_centroid(v_r,1)))
				count = count + 1;
				cut = 1;
			elseif (cut)
				break; 
			end
		end
		count = count - px_border_corr;
		v_px_len = count/v_r_len;

		% determine width of horizontal rectangle (take row passing through h rect
		% and count number of white pixels in it
		count = sum(QR_rect_local(round(mask_centroid(h_r,2)),:))-px_border_corr;
		h_px_len = count/h_r_len;

		% find top right corner of QR code
		px1 = 5; % # of QR tiles from vert rect centroid to top-right corner centroid
		loc_tr_row = mask_centroid(v_r,2) - px1*v_px_len;
		dist_diff = abs(QR_centroid - [mask_centroid(v_r,1), loc_tr_row]);
		dist_diff = dist_diff(:,1)+dist_diff(:,2);
		[~, TR_index] = min(dist_diff);
		TR_coord = [QR_centroid(TR_index,2) QR_centroid(TR_index,1)];

		% find bottom left corner of QR code
		px2 = 8; % # of QR tiles from horiz rect centroid to bot-left corner centroid
		loc_bl_col = mask_centroid(h_r,1) - px2*h_px_len;
		dist_diff = abs(QR_centroid - [loc_bl_col, mask_centroid(h_r,2)]);
		dist_diff = dist_diff(:,1)+dist_diff(:,2);
		[~, BL_index] = min(dist_diff);
		BL_coord = [QR_centroid(BL_index,2) QR_centroid(BL_index,1)];

		% find bottom right corner of QR code
		dist_diff = abs(QR_centroid - [mask_centroid(h_r,1) mask_centroid(h_r,2)]);
		dist_diff = dist_diff(:,1)+dist_diff(:,2);
		[~, h_r_index] = min(dist_diff);
		dist_diff = abs(QR_centroid - [mask_centroid(v_r,1) mask_centroid(v_r,2)]);
		dist_diff = dist_diff(:,1)+dist_diff(:,2);
		[~, v_r_index] = min(dist_diff);
		BR_coord =[QR_centroid(h_r_index,2) QR_centroid(v_r_index,1)];
		
		% find top left corner of QR code
		% var for shifting expected location of top-left corner (expecting shearing)
		row_corr = -5;
		col_corr = -5;
		dist_diff = abs(QR_centroid - [BL_coord(2)+col_corr, TR_coord(1)+row_corr]);
		dist_diff = dist_diff(:,1)+dist_diff(:,2);
		[~, TL_index] = min(dist_diff);
		TL_coord = [QR_centroid(TL_index,2) QR_centroid(TL_index,1)];

		% create mapping of coordinates to QR code while accounting for distortion
		steps = 10; % 10 steps if starting at one corner and moving another corner
		num_px = 11;
		y_coords = zeros(num_px,num_px);
		x_coords = zeros(num_px,num_px);
		y0 = TL_coord(1); x0 = TL_coord(2); 
		ym = BR_coord(1); xm = BR_coord(2);
		
		% column information to match any stretching in QR code
		y1 = TR_coord(1); x1 = TR_coord(2); 
		yn = BL_coord(1); xn = BL_coord(2);
		y_slope = (y1 - y0) / (x1 - x0); y_l_slope = (ym - yn) / (xm - xn);
		x_steps = (x1 - x0) / steps; x_l_steps= (xm - xn) / steps;
		for i = 1:num_px
			y = y0 + y_slope * ((i - 1) * x_steps);
			y_B = yn + y_l_slope * ((i - 1) * x_l_steps);
			v_px_height = (y_B - y) / steps;
			y_coords(1:num_px,i) = y:v_px_height:y_B';
		end

		% row imformation to match and stretching in QR code
		x1 = BL_coord(2); y1 = BL_coord(1); 
		yn = TR_coord(1); xn = TR_coord(2);
		x_slope = (x1 - x0) / (y1 - y0); x_l_slope = (xm - xn) / (ym - yn);
		y_steps = (y1 - y0) / steps; y_l_steps = (ym - yn) / steps;
		for i = 1:num_px
			x = x0 + x_slope * ((i - 1) * y_steps);
			x_R = xn + x_l_slope * ((i - 1) * y_l_steps);
			h_px_height = (x_R - x) / steps;
			x_coords(i, 1:num_px) = x:h_px_height:x_R;
		end

		% use warped row and col coords to read QR code
		w_pad = 1;
		QR_val = zeros(num_px,num_px);
		for row = 1:num_px
			for col = 1:num_px
				%select 3x3 area centred around where expected QR segment centre is
				window = QR_cleaned_adj(round(y_coords(row,col) - w_pad:y_coords(row,col) + w_pad), round(x_coords(row,col) - w_pad:x_coords(row,col) + w_pad));
				median_val = median(window, 'all');
				QR_val(row,col) = median_val;

				% shows where each pixel value was taken from on QR code
				QR_cleaned_adj(round(y_coords(row,col)), round(x_coords(row,col))) = ~QR_val(row,col);
			end
		end

		% get information out of QR code
		[data_recovered, str_recovered] = interpret(QR_val);
		QR_codes_data{1,img_QR_index} = data_recovered;
		
		% display QR code
		if (show_images == 1)
			figure(img_QR_index);
			subplot(2,2,1);	
			imshow(QR_codes{img_QR_index});	
			title(sprintf('Image #%d, QR code #%d',image_num, img_QR_index));
			
			subplot(2,2,2);	
			imshow(QR_cleaned_adj);
			
			subplot(2,2,3);	
			imagesc(~QR_val); 		
			colormap gray; 
			axis equal; 
			axis off;
			title(['value = ', str_recovered]);
			impixelinfo;
		end
	end % end of all QR codes

	% show original image with QR codes marked
	 if (show_images == 1)
		 figure; imshow(img_QR_index_original);hold on;
		 for img_QR_index = 1:length(QR_codes_data);
			plot(QR_coords{img_QR_index}(1),QR_coords{img_QR_index}(2), 'r+', 'MarkerSize', 30, 'LineWidth', 2);
		end
	 end

	data_out = QR_codes_data;
	coord_out = QR_coords;

end % end of function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simple_th - highlight values if they fall below th_val, otherwise do not change image
% @param I_in - input image to apply threshold to
% @param th_val - threshold value to use
% @return out - thresholded image
function out = simple_th(I_in, th_val)
    [M,N] = size(I_in);
    out = uint8(zeros(M,N));
    for row = 1:M
        for col = 1:N
            if (I_in(row,col) <= th_val)
                out(row,col) = 255;
            else
                out(row,col) = 0;
            end
        end
    end
end

% booster - increase contrast of edges of image
% @param I_in - image to apply edge contrast of
% @return out - image with boosted edge contrast
function out = booster(I_in)
    % control variables    
    filt_size = 7;
    img_QR_indexhigh_f_coef = 255;
    img_QR_indexlow_f_coef = 0;

    % high-boost (get high frequency, add to original)
    filt = fspecial('average', filt_size);
    img_QR_indexlow_f = imfilter(I_in, filt, 'symmetric');
    img_QR_indexhigh_f = I_in-img_QR_indexlow_f;

    out = I_in + img_QR_indexhigh_f_coef*img_QR_indexhigh_f - img_QR_indexlow_f_coef*img_QR_indexlow_f;
end

% rgb_thresh - threshold rgb image to highlight values below bk_thresh
% @param I_in - image to apply threshold to
% @return out - image with applied threshold
function out = rgb_thresh(I_in)
    [M,N,D] = size(I_in);
    int_im = zeros(M,N);
    bk_thresh = 145; % im_8:80 ; 128; 50
    
    for row = 1:M
        for col = 1:N
            % to turn into black and white
            value = sum(I_in(row,col,:))/(3);
            if (value < bk_thresh)
                int_im(row,col) = 255;
            else
                int_im(row,col) = 0;
            end
        end
    end
    out = uint8(int_im);
end

% gray_thresh - threshold grayscale image
% @param I_in - image to apply threshold to
% @return out - image with applied threshold
function out = gray_thresh(I_in)
    [M,N] = size(I_in);
    int_im = zeros(M,N);
    bk_thresh = 200; % control variable
    
    for row = 1:M
        for col = 1:N
            % to turn into black and white
            value = I_in(row,col);
            if (value < bk_thresh)
                int_im(row,col) = 255;
            else
                int_im(row,col) = 0;
            end
        end
    end
    out = uint8(int_im);
end

% th - threshold image based on pixels highlighted in inputted mask
% @param I_in - input image to threshold
% @param mask - mask to threshold image with
% @return out - thresholded image
function out = th(I_in, mask)
rect_area = uint8(I_in+1) .* uint8(mask);
[counts, X] = imhist(rect_area);
average_coeff = 1.5;
average = round(average_coeff * (1 / sum(counts(2:length(counts)))) * sum(counts(2:length(counts)) .* X(1:(length(X) - 1))));

    [img_rows, img_cols] = size(I_in);
    out = I_in;
    for row = 1:img_rows
        for col = 1:img_cols
            if (I_in(row,col) < average)
                out(row,col) = 255;
            else
                out(row,col) = 0;
            end
        end
    end
end

% cleanup - some operations used to remove blobs from post-thresholded QR sub-image
% @param I_in - imput image to cleanup blobs of
% @return out - image after being cleaned up
function out = cleanup(I_in)
    % term for filtering noise
    min_px_size_wt = 11;
    min_px_size_bk = 15;
    
    % remove blobs touching border
    QR_adj = imclearborder(I_in);
    
    % remove blobs not a certain size
    QR_adj = bwareafilt(logical(QR_adj), [min_px_size_wt, inf]);
    QR_adj = imdilate(QR_adj, strel('square', 2));
    QR_adj = ~bwareafilt(logical(~QR_adj), [min_px_size_bk, inf]);
    QR_adj = ~imdilate(~QR_adj, strel('square', 2));
    
    % smooth edges and fill in small holes
    [img_rows, img_cols] = size(I_in);    
    w_pad = 1; % size of padding of search window
    thresh_blob_size = 5; % if 5+ pixels are 0, make this pixel white too
    
    for row = 1 + w_pad:img_rows - w_pad
        for col = 1 + w_pad:img_cols - w_pad
            w_ = ~QR_adj(row - w_pad:row + w_pad, col - w_pad:col + w_pad);
            if sum(logical(w_), 'all') >= thresh_blob_size;
                QR_adj(row, col) = 0;
            end
        end
    end
    out = QR_adj;
end

% interpret - function to get encoded information out of QR code
% 	outputs that information in terms of a 1x83 array and a string
% @param data - QR code data to decode
% @return bin_out - QR decoded data in binary form
% @return str_out - QR decoded data in string form
function [bin_out, str_out] = interpret(data)
    % NOTE: data in QR code is inputted by column (top-to-bottom of column,
    % leftmost col to rightmost col) instead of by row (left-to-right of 
    % row, topmost row to bottommost row) so interpret data in the same way
    
    data = logical(data);
    max_data = 83;
    null_char = '0'; % character separating a string from a character array
    code_mask = [ ...
        1 0 2 2 2 2 2 2 2 0 1; ...
        0 0 2 2 2 2 2 2 2 0 0; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        2 2 2 2 2 2 2 2 2 0 1; ...
        0 0 2 2 2 0 0 0 0 0 0; ...
        1 0 2 2 2 0 1 1 1 1 1]; 
    bin_out = zeros(1,max_data);
    [QR_row,QR_col] = size(code_mask);
    count = 1;
    
    % go through every pixel in inputted data
    for col = 1:QR_col
        for row = 1:QR_row
            % if data on certain part of QR code, record that data
            if (code_mask(row,col) == 2)
                bin_out(count) = data(row,col);
                count = count+1;
            end
        end
    end
    
    % turn binary matrix into string
    char_stream = char(bin_out(1, 1:80) + null_char); 
    
    % break up string into byte (character) matrix
    characters = reshape(char_stream, 8, []).'; 
    
    % convert binary to decimal, then decimal into (ASCII) characters
    str_out = char(bin2dec(characters).');
end