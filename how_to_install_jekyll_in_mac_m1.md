# How to install jekyll in rbenv using homebrew in MacOS Silicon Chip
### Install Homebrew
    $ su ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

### Show brew commands
    $ brew help

### Check current user
    $ echo $(whoami)

### Grant access to the folders
    $ sudo chown -R $(whoami) /usr/local
    $ sudo chown -R $(whoami) /Library/Caches/Homebrew/

### Uninstall brew ruby
    $ brew uninstall ruby

### Install rbenv
    $ brew update
    $ brew install rbenv ruby-build

### Add ~/.rbenv/bin to your $PATH for access to the rbenv command-line utility
    $ echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bash_profile

### Add rbenv init to your shell to enable shims and autocompletion
    $ echo 'eval "$(rbenv init -)"' >> ~/.bash_profile

### Install ruby via rbenv
    $ rbenv install 2.2.3
    $ rbenv global 2.2.3
    $ rbenv versions

### Check install path
    $ which ruby
    $ which gem

### Rehash
    $ rbenv rehash

### Check ruby version and environment
    $ ruby --version
    $ gem env

### Install bundler
    $ gem install bundler

### Go to the project folder
    $ cd <project folder>

### Install / update gems
    $ bundle install
      or
    $ bundle update

### Show installed jekyll
    $ bundle show jekyll

### Serve jekyll pages
    $ bundle exec jekyll serve