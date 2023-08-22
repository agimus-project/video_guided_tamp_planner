import os
import sys
import time
import subprocess
from hpp.corbaserver.manipulation import Client, loadServerPlugin
from omniORB.CORBA import SystemException


class CorbaServer:
    def __init__(self, models_package=None, start=True) -> None:
        super().__init__()
        self.process = None
        self.models_package = models_package
        if start:
            self.start()

    def start(self):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        ld_path = f"LD_LIBRARY_PATH={conda_prefix}/lib"
        ros_pkg_path = f"ROS_PACKAGE_PATH={conda_prefix}/share/:{self.models_package}/"
        corba_command = f"{ld_path} {ros_pkg_path} hppcorbaserver"

        self.process = subprocess.Popen(corba_command, shell=True)
        print(corba_command)

        assert self.wait_for_corba(), "ERROR: time out, could not start corba server!"

        Client().problem.resetProblem()
        return self

    def reset_problem(self):
        Client().problem.resetProblem()

    @staticmethod
    def wait_for_corba(time_out=100):
        """
        will wait for corba to properly boot up

        :param time_out: how many 0.05s should we wait for corba before time out
        :return: True if success, false if time out
        """

        def block_print():
            sys.stdout = open(os.devnull, "w")

        def enable_print():
            sys.stdout = sys.__stdout__

        block_print()
        for i in range(time_out):
            try:
                loadServerPlugin(
                    "corbaserver",
                    os.environ.get("CONDA_PREFIX")
                    + "/lib/hppPlugins/manipulation-corba.so",
                )
                enable_print()
                return True
            except SystemException:
                time.sleep(0.05)
            except Exception as e:
                print(e)

        enable_print()
        return False

    def kill(self):
        corba_command = "killall hppcorbaserver"
        p = subprocess.Popen(corba_command, shell=True)
        p.wait()
        self.process.wait()

    def __del__(self):
        self.kill()

    def restart(self):
        self.kill()
        time.sleep(1)
        self.start()
