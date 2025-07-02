git config --file .gitmodules --get-regexp path | awk '{ print $2 }' | while read line; do
    echo "Removing $line"
    # Deinit the submodule
    git submodule deinit -f "$line"
    
    # Remove the submodule from the working tree
    git rm -f "$line"
    
    # Remove the submodule from .git/modules
    rm -rf ".git/modules/$line"
done

# Remove the .gitmodules file
git rm -f .gitmodules
