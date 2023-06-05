import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class ImageBrowser:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Browser")
        # 设置GUI的布局
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(fill="both", expand=True)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill="both", expand=True)

        # 创建底部框架
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        # 创建状态栏标签
        self.status_label = tk.Label(self.bottom_frame, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(fill="x", padx=10, pady=10)

        self.skip_button = tk.Button(self.button_frame, text="Skip(→)", command=self.skip_image)
        self.skip_button.pack(side="left", padx=10)

        self.save_button = tk.Button(self.button_frame, text="Save(↓)", command=self.save_image)
        self.save_button.pack(side="left", padx=10)

        self.input_button = tk.Button(self.button_frame, text="Input", command=self.set_input_directory)
        self.input_button.pack(side="left", padx=10)

        self.output_button = tk.Button(self.button_frame, text="Output", command=self.set_output_directory)
        self.output_button.pack(side="left", padx=10)


        # 初始化变量
        self.input_directory = filedialog.askdirectory(title="Select Input Directory")
        self.output_directory = filedialog.askdirectory(title="Select Output Directory")
        self.image_list = []
        self.current_image = 0
        self.num_saved = 0
        
        # 读取图片
        self.load_images()

        # 显示第一张图片
        self.show_image()

        # 监听键盘事件
        self.master.bind("<Right>", lambda event: self.skip_image())
        self.master.bind("<Down>", lambda event: self.save_image())

    def load_images(self):
        # 弹出文件选择窗口选择输入目录

        # 遍历目录中的所有文件，如果是图像文件则添加到图片列表中
        for file in os.listdir(self.input_directory):
            filepath = os.path.join(self.input_directory, file)
            if os.path.isfile(filepath) and file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                self.image_list.append(filepath)

    def show_image(self):
        # 获取当前图片路径
        image_path = self.image_list[self.current_image]

        # 打开并缩放图片
        image = Image.open(image_path)
        # image = image.resize((self.image_frame.winfo_width(), self.image_frame.winfo_height()))

        # 将图片转换为Tkinter对象并显示
        tk_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image
        
        
        # 更新状态栏
        num_saved = self.num_saved
        status_text = f"Image {self.current_image + 1}/{len(self.image_list)} ({num_saved}/{len(self.image_list)} saved): {os.path.basename(self.image_list[self.current_image])}"
        self.status_label.config(text=status_text)

    def skip_image(self):
        # 切换到下一张图片
        self.current_image += 1
        if self.current_image >= len(self.image_list):
            self.current_image = 0

        # 显示图片
        self.show_image()

    def save_image(self):
        # 如果未设置输出目录，则提示用户设置输出目录
        if not self.output_directory:
            self.set_output_directory()
            return

        # 获取当前图片路径和输出路径
        image_path = self.image_list[self.current_image]
        output_path = os.path.join(self.output_directory, os.path.basename(image_path))

        # 将图片复制到输出目录
        shutil.copy(image_path, output_path)
        
        # 更新状态栏
        self.num_saved+=1
        num_saved = self.num_saved
        status_text = f"Image {self.current_image + 1}/{len(self.image_list)} ({num_saved}/{len(self.image_list)} saved): {os.path.basename(self.image_list[self.current_image])}"
        self.status_label.config(text=status_text)
        # 显示下一张图片
        self.skip_image()

    def set_input_directory(self):
        # 弹出文件选择窗口选择输入目录
        self.input_directory = filedialog.askdirectory(title="Select Input Directory")

        # 重新加载图片
        self.image_list = []
        self.current_image = 0
        self.load_images()
        self.show_image()

    def set_output_directory(self):
        # 弹出文件选择窗口选择输出目录
        self.output_directory = filedialog.askdirectory(title="Select Output Directory")

root = tk.Tk()
app = ImageBrowser(root)
root.mainloop()