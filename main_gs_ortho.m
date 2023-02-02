A = eye(3);
W = eye(3);
B1 = gs_ortho(A,W);
disp('The result of the first test case is:');
disp(B1);

A2 = [1 2; 1 1];
W2 = eye(2);
B2 = gs_ortho(A2,W2);
disp('The result of the second test case is:');
disp(B2);

A3 = [1 1 2; 2 1 1; 1 3 1];
W3 = [2 -1 0; -1 2 -1; 0 -1 2];
B3 = gs_ortho(A3,W3);
disp('The result of the third test case is:');
disp(B3);
disp('Here are the calculations of the square root terms: 1/sqrt(2), (1/2)/sqrt(13), and (5/2)/sqrt(13)');
disp(1/sqrt(2));
disp((1/2)/sqrt(13));
disp((5/2)/sqrt(13));
