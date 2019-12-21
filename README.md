Workflow:
For R or local preview
1. Open website R project
2. Click Serve site, it only run new R files have just been added
3. Add R file to its correct folder project/post

For Python
1. copy jupyter notebook to the correct folder project/post
2. initialize index.md with correct header
3. Convert notebook to markdown (jupyter notebook file must be in the same folder) by jupyter nbconvert Untitled.ipynb --to markdown --NbConvertApp.output_files_dir=.
4. Copy generated markdown to the index.md file, and delete unnecessary files
