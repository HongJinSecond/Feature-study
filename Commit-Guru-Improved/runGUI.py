import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import io
from tkinter import font as tkfont
import ctypes
from offline import offline_process as process


class ProcessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Expert Feature Extraction")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        
        # 设置字体
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.default_font.configure(size=9)
        self.text_font = tkfont.Font(family="Arial", size=9)
        self.title_font = tkfont.Font(family="Arial", size=16, weight="bold")
        self.subtitle_font = tkfont.Font(family="Arial", size=10)
        self.header_font = tkfont.Font(family="Arial", size=10, weight="bold")
        self.status_font = tkfont.Font(family="Arial", size=10, weight="bold")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 配置样式
        self.style.configure('Title.TLabel', 
                            font=self.title_font,
                            background='#f0f0f0',
                            foreground='#333333')
        
        self.style.configure('Subtitle.TLabel',
                            font=self.subtitle_font,
                            background='#f0f0f0',
                            foreground='#666666')
        
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=self.text_font)
        self.style.configure('TLabel', background='#f0f0f0', font=self.text_font)
        self.style.configure('Header.TLabel', font=self.header_font)
        self.style.configure('TLabelframe', font=self.header_font)
        self.style.configure('TLabelframe.Label', font=self.header_font)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20", style='TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(self.main_frame, text="Expert Feature Extraction", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = ttk.Label(self.main_frame, 
                                  text="Please input the params and click start button.", 
                                  style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # 输入框架
        input_frame = ttk.LabelFrame(self.main_frame, text="Input Params", padding="15")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 目标名称输入
        ttk.Label(input_frame, text="Project Name:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(input_frame, textvariable=self.name_var, width=30, font=self.text_font)
        self.name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # 模式选择
        ttk.Label(input_frame, text="Mode:", style='Header.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar()
        self.mode_combo = ttk.Combobox(input_frame, textvariable=self.mode_var, 
                                      state="readonly", width=28, font=self.text_font)
        self.mode_combo['values'] = ('New', 'Old')
        self.mode_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        self.mode_combo.current(0)
        
        # 处理按钮
        self.process_btn = ttk.Button(self.main_frame, text="start", 
                                     command=self.start_processing, width=20)
        self.process_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        # 状态显示
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(status_frame, text="Stats:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W)
        self.status_var = tk.StringVar(value="Wating for input")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     foreground="#007acc", font=self.status_font)
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(self.main_frame, text="Process result.", padding="15")
        result_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        ttk.Label(result_frame, text="Output path:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.result_var = tk.StringVar()
        self.result_entry = ttk.Entry(result_frame, textvariable=self.result_var, 
                                     state="readonly", width=40, font=self.text_font)
        self.result_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # 错误信息显示框架
        error_frame = ttk.LabelFrame(self.main_frame, text="log", padding="15")
        error_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.error_text = tk.Text(error_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, font=self.text_font)
        error_scrollbar = ttk.Scrollbar(error_frame, orient=tk.VERTICAL, command=self.error_text.yview)
        self.error_text.configure(yscrollcommand=error_scrollbar.set)
        
        self.error_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        error_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置网格权重
        input_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(1, weight=1)
        error_frame.columnconfigure(0, weight=1)
        error_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(6, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 绑定回车键到开始处理
        self.root.bind('<Return>', lambda event: self.start_processing())
        
        # 设置初始焦点
        self.name_entry.focus()
    
    def start_processing(self):
        # 获取输入参数
        name = self.name_var.get()
        mode = self.mode_var.get()
        
        # 验证输入
        if not name or not mode:
            messagebox.showerror("输入错误 Error Input", "请填写目标名称和选择模式！")
            return
        
        # 禁用按钮并更新状态
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("正在处理...")
        self.status_label.configure(foreground="#007acc")
        self.result_var.set("")
        self.error_text.config(state=tk.NORMAL)
        self.error_text.delete(1.0, tk.END)
        self.error_text.insert(tk.END, f"开始处理: 名称={name}, 模式={mode}\n")
        self.error_text.see(tk.END)
        self.error_text.config(state=tk.DISABLED)
        
        # 在新线程中运行处理过程
        thread = threading.Thread(target=self.run_process, args=(name, mode))
        thread.daemon = True
        thread.start()
        
        # 定期检查线程状态
        self.check_thread(thread)
    
    def run_process(self, name, mode):
        """在新线程中运行处理函数"""
        try:
            # 重定向标准输出和错误输出以捕获错误信息
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            captured_output = io.StringIO()
            sys.stdout = captured_output
            sys.stderr = captured_output
            
            # 调用处理函数 - 注意参数顺序调整为(name, mode)
            result = process(name, mode)
            
            # 恢复标准输出和错误输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # 将结果和可能的输出信息传递回主线程
            self.root.after(0, self.on_process_success, result, captured_output.getvalue())
            
        except Exception as e:
            # 恢复标准输出和错误输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # 将错误信息传递回主线程
            error_msg = f"{type(e).__name__}: {str(e)}"
            output_msg = captured_output.getvalue()
            self.root.after(0, self.on_process_error, error_msg, output_msg)
    
    def check_thread(self, thread):
        """检查线程是否完成"""
        if thread.is_alive():
            # 线程仍在运行，100毫秒后再次检查
            self.root.after(100, self.check_thread, thread)
        else:
            # 线程已完成，但结果已通过其他方式处理
            pass
    
    def on_process_success(self, result, output):
        """处理成功时的回调函数"""
        self.status_var.set("Process success!")
        self.status_label.configure(foreground="#4caf50")  # 绿色表示成功
        self.result_var.set(result)
        self.process_btn.config(state=tk.NORMAL)
        
        # 显示输出信息
        if output:
            self.error_text.config(state=tk.NORMAL)
            self.error_text.insert(tk.END, f"\Output:\n{output}")
            self.error_text.see(tk.END)
            self.error_text.config(state=tk.DISABLED)
        
        # 添加成功日志
        self.error_text.config(state=tk.NORMAL)
        self.error_text.insert(tk.END, f"\Output path: {result}")
        self.error_text.see(tk.END)
        self.error_text.config(state=tk.DISABLED)
    
    def on_process_error(self, error_msg, output_msg):
        """处理错误时的回调函数"""
        self.status_var.set("Process failed!")
        self.status_label.configure(foreground="#f44336")  # 红色表示错误
        self.process_btn.config(state=tk.NORMAL)
        
        # 显示错误信息
        self.error_text.config(state=tk.NORMAL)
        self.error_text.insert(tk.END, f"\n错误: {error_msg}\n")
        
        # 添加任何输出信息
        if output_msg:
            self.error_text.insert(tk.END, f"\n输出:\n{output_msg}")
        
        self.error_text.see(tk.END)
        self.error_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessApp(root)
    root.mainloop()