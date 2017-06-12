# Setup

The code is written in *Python 2.7* against *OpenCV 2.4*.  Install
[OpenCV](http://opencv.org/) for your platform and this should bring in the
only other dependency, `numpy`.

1. ### Download spot_larva and its library

    You'll need to obtain both the script `spot_larva.py` and the `lib/`
    directory.  The easiest way to do that is with `git`, in a shell
    interpreter (eg. `bash`).

    ```sh-session
    $ git clone https://github.com/plredmond/larva-tracker.git
    ```

    This command will create a folder in your current working directory called
    `larva-tracker` which has the the code repository.

1. ### Install OpenCV for your platform

    1. #### Mac OSX

        Use Mac OSX [homebrew](http://brew.sh/) to install the `opencv`
        package.

        ```sh-session
        $ brew install homebrew/science/opencv
        ```

    1. #### Arch Linux

        Use [pacman](https://wiki.archlinux.org/index.php/Pacman) to install the
        [opencv](https://www.archlinux.org/packages/extra/x86_64/opencv/) and
        [numpy](https://www.archlinux.org/packages/extra/x86_64/python2-numpy/)
        packages.

        ```sh-session
        $ pacman -Syu opencv python2-numpy
        ```
