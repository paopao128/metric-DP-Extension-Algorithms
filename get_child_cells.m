function [child_cells_points, child_weights] = get_child_cells(points)
    % get_child_cells Refines a spatial region into sub-cells for mDP tree extension
    % Input: 
    %   points - Either the global anchor points (N x 2) or a single cell's corners (4 x 2)
    % Output:
    %   child_cells_points - Cell array of coordinates for each child cell (1 x M)
    %   child_weights      - Cell array of sparse weight matrices W mapping child to parent (1 x M)
    
    k = 2; % Encryption multiplier (k=2 means 2x2 = 4 subcells per parent)
    nOrig = size(points, 1);
    
    child_cells_points = {};
    child_weights = {};
    
    xUnique = sort(unique(points(:,1)));
    yUnique = sort(unique(points(:,2)));
    
    if nOrig > 4
        % -----------------------------------------------------------------
        % SCENARIO 1: ROOT NODE (Global Grid)
        % Break the entire map down into individual coarse base cells.
        % -----------------------------------------------------------------
        for i = 1:length(yUnique)-1
            for j = 1:length(xUnique)-1
                x0 = xUnique(j); x1 = xUnique(j+1);
                y0 = yUnique(i); y1 = yUnique(i+1);
                
                % Find indices of the 4 corners in the parent 'points' array
                idx_00 = find(points(:,1)==x0 & points(:,2)==y0);
                idx_10 = find(points(:,1)==x1 & points(:,2)==y0);
                idx_01 = find(points(:,1)==x0 & points(:,2)==y1);
                idx_11 = find(points(:,1)==x1 & points(:,2)==y1);
                
                % Skip incomplete cells (e.g., map boundaries not perfectly square)
                if isempty(idx_00) || isempty(idx_10) || isempty(idx_01) || isempty(idx_11)
                    continue; 
                end
                
                % Define the 4 corners of this specific base cell
                child_points = [x0, y0; x1, y0; x0, y1; x1, y1];
                
                % The weight matrix simply selects these 4 points from the global array
                W = sparse(4, nOrig);
                W(1, idx_00) = 1;
                W(2, idx_10) = 1;
                W(3, idx_01) = 1;
                W(4, idx_11) = 1;
                
                child_cells_points{end+1} = child_points;
                child_weights{end+1} = W;
            end
        end
        
    else
        % -----------------------------------------------------------------
        % SCENARIO 2: RECURSIVE NODE (Single Cell)
        % Split a single 4-corner cell into k*k subcells.
        % -----------------------------------------------------------------
        x0 = min(points(:,1)); x1 = max(points(:,1));
        y0 = min(points(:,2)); y1 = max(points(:,2));
        
        xAll = linspace(x0, x1, k+1);
        yAll = linspace(y0, y1, k+1);
        
        for i = 1:k
            for j = 1:k
                sub_x0 = xAll(j); sub_x1 = xAll(j+1);
                sub_y0 = yAll(i); sub_y1 = yAll(i+1);
                
                % Define the 4 corners of the new subcell
                child_points = [sub_x0, sub_y0; sub_x1, sub_y0; sub_x0, sub_y1; sub_x1, sub_y1];
                
                % Bilinear interpolation weights relative to the parent cell
                W = sparse(4, 4);
                for pt_idx = 1:4
                    px = child_points(pt_idx, 1);
                    py = child_points(pt_idx, 2);
                    
                    if x1 ~= x0
                        lx = (px - x0) / (x1 - x0);
                    else
                        lx = 0;
                    end
                    
                    if y1 ~= y0
                        ly = (py - y0) / (y1 - y0);
                    else
                        ly = 0;
                    end
                    
                    % Map relative to parent ordering: 1:(x0,y0), 2:(x1,y0), 3:(x0,y1), 4:(x1,y1)
                    W(pt_idx, 1) = (1 - lx) * (1 - ly);
                    W(pt_idx, 2) = lx       * (1 - ly);
                    W(pt_idx, 3) = (1 - lx) * ly;
                    W(pt_idx, 4) = lx       * ly;
                end
                
                child_cells_points{end+1} = child_points;
                child_weights{end+1} = W;
            end
        end
    end
end