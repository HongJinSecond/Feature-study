import time
import tracemalloc
import numpy as np
from offline import offline_process
import pandas as pd


project_list = ["brackets","BroadleafCommerce","camel","corefx-v-6.2","django","elasticsearch",
"fabric8","godot","Mybatis3","mysql-server","node","nova",
"npm","pandas","pytorch","rails","rust","security","spring-boot","tomcat","vscode","wp-calypso"
]


modes = ["New","Old"]

Sample = 1000



for mode in modes:
    table = []
    for project in project_list:
        print(f"当前执行项目{project}--模式：{mode}")
        tracemalloc.start()
        start = time.perf_counter()

        test = np.random.random((1000,1000))


        offline_process(project,mode,Sample,Auto_Labeling=False)



        current, peak = tracemalloc.get_traced_memory()
        print(f"当前内存: {current / 10**6:.1f} MB, 峰值内存: {peak / 10**6:.1f} MB")
        tracemalloc.stop()
        end = time.perf_counter()
        print(f"运行耗时: {end - start:.4f} 秒")

        table.append([end - start,peak / 10**6])
    print("Save to csv")

    pd.DataFrame(data=table,columns=["time","memory"],index=project_list).to_csv(f"{mode}_overhead.csv",encoding="utf-8")


