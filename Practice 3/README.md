# Практическая работа №3 
## Параллельная реализация простых алгоритмов сортировки с OpenMP

**Student: Aruzhan Imasheva**

**Group: ADA-2404M**
---

# 📑 Содержание


<pre>
graph TD
    A[Start: Host Memory] --&gt; B[Initialize Random Data]
    B --&gt; C[CUDA Malloc]
    C --&gt; D[Kernel: blockSort]
    D --&gt; E[Kernel: mergeKernel]
    E --&gt; F[DeviceToHost Memcpy]
    F --&gt; G[End]
</pre>

