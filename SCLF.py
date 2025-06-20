import numpy as np
import time
import matplotlib.pyplot as plt
import pdb

np.set_printoptions(
    ##threshold = 1000000
    formatter={'float': '{:+7.2f}'.format},  # 每个浮点数占 8 格、保留 3 位小数
    linewidth=800,                           # 每行最多 200 字符，尽量打印在一行内
    suppress=True                            # 小数点后的小于 1e-4 的数字不使用科学计数法
)

PAD = 1e9
time_duration = 1
## SCL decoding in this python file ##
from functools import lru_cache

'''
    N = code length
    M = message length
    r = CRC
    K = information length = M + r
    N - K = frozen length
    R = coderate = K/N
    epsilon = cross probability
'''
@lru_cache(maxsize=None)



def bit_reverse(i, n):
    rev = 0
    for j in range(n):
        rev <<= 1
        rev |= (i >> j) & 1
    return rev

def f_func(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

def g_func(a, b, u):
    return b + (1 - 2*u) * a

class Path:
    def __init__(self, llr_tree, bit_tree, pm):
        self.llr_tree = [lvl.copy() for lvl in llr_tree]
        self.bit_tree = [lvl.copy() for lvl in bit_tree]
        self.pm = pm
    def copy(self):
        return Path(self.llr_tree, self.bit_tree, self.pm)
        
class polar_code:
    def __init__(self, cross_p, N, R, CRC, List, O, T_flip, Alpha):
        self.cross_p = cross_p
        self.N = int(N)
        self.n = int(np.log2(self.N))
        self.R = R
        self.M = int(np.ceil(N*R))
        self.CRC = int(CRC)
        self.K = int(self.M + self.CRC)
        self.List = int(List)
        self.Order = int(O)
        self.T_flip = int(T_flip)
        self.Alpha = Alpha

        self.cal_bha()
        self.Info_Bha()
        #self.InfoBitSelectPW()
        self.Generator_Polar()
        if(self.CRC>0):
            self.crc_generator_matrix()
        self.book()


    def cal_bha(self):
        self.bha_list = np.ones(self.N)*self.cross_p
        for i in range(self.N):
            bha_weight = np.zeros(self.n)
            tmp = i
            for j in range(self.n):
                bha_weight[j] = tmp % 2
                tmp = tmp >> 1
            for j in range(self.n):
                bha_tmp = self.bha_list[i]
                if bha_weight[(self.n-1) - j] == 1:
                    self.bha_list[i] = pow(bha_tmp, 2)
                else:
                    self.bha_list[i] = 2*bha_tmp - pow(bha_tmp, 2)

    def Info_Bha(self):
        sort = []
        self.frozen_set = []
        for i in range(self.N):
            sort.append(self.bha_list[i])

        sort = np.array(sort)
        sort = np.sort(sort, axis=0)
        threshold = sort[int(self.K - 1)]
        for i in range(self.N):
            if(self.bha_list[i]>threshold):
                self.frozen_set.append(i)
        self.info_set = [i for i in range(self.N) if i not in self.frozen_set]

    def InfoBitSelectPW(self):
        if self.N == 1024:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,64	,9	,6	,17	,10	,18	,128	,12	,33	,65	,20	,256	,34	,24	,36	,7	,129	,66	,512	,11	,40	,68	,130	,19	,13	,48	,14	,72	,257	,21	,132	,35	,258	,26	,513	,80	,37	,25	,22	,136	,260	,264	,38	,514	,96	,67	,41	,144	,28	,69	,42	,516	,49	,74	,272	,160	,520	,288	,528	,192	,544	,70	,44	,131	,81	,50	,73	,15	,320	,133	,52	,23	,134	,384	,76	,137	,82	,56	,27	,97	,39	,259	,84	,138	,145	,261	,29	,43	,98	,515	,88	,140	,30	,146	,71	,262	,265	,161	,576	,45	,100	,640	,51	,148	,46	,75	,266	,273	,517	,104	,162	,53	,193	,152	,77	,164	,768	,268	,274,518	,54	,83	,57	,521	,112	,135	,78	,289	,194	,85	,276	,522	,58	,168	,139	,99	,86	,60	,280	,89	,290	,529	,524	,196	,141	,101	,147	,176	,142	,530	,321	,31	,200	,90	,545	,292	,322	,532	,263	,149	,102	,105	,304	,296	,163	,92	,47	,267	,385	,546	,324	,208	,386	,150	,153	,165	,106	,55	,328	,536	,577	,548	,113	,154	,79	,269	,108	,578	,224	,166	,519	,552	,195	,270	,641	,523	,275	,580	,291	,59	,169	,560	,114	,277	,156	,87	,197	,116	,170	,61	,531	,525	,642	,281	,278	,526	,177	,293	,388	,91	,584	,769	,198	,172	,120	,201	,336	,62	,282	,143	,103	,178	,294	,93	,644	,202	,592	,323	,392	,297	,770	,107	,180	,151	,209	,284	,648,94	,204	,298	,400	,608	,352	,325	,533	,155	,210	,305	,547	,300	,109	,184	,534	,537	,115	,167	,225	,326	,306	,772	,157	,656	,329	,110	,117	,212	,171	,776	,330	,226	,549	,538	,387	,308	,216	,416	,271	,279	,158	,337	,550	,672	,118	,332	,579	,540	,389	,173	,121	,553	,199	,784	,179	,228	,338	,312	,704	,390	,174	,554	,581	,393	,283	,122	,448	,353	,561	,203	,63	,340	,394	,527	,582	,556	,181	,295	,285	,232	,124	,205	,182	,643	,562	,286	,585	,299	,354	,211	,401	,185	,396	,344	,586	,645	,593	,535	,240	,206	,95	,327	,564	,800	,402	,356	,307	,301	,417	,213	,568	,832	,588	,186	,646	,404	,227	,896	,594	,418	,302	,649	,771	,360	,539	,111	,331,214	,309	,188	,449	,217	,408	,609	,596	,551	,650	,229	,159	,420	,310	,541	,773	,610	,657	,333	,119	,600	,339	,218	,368	,652	,230	,391	,313	,450	,542	,334	,233	,555	,774	,175	,123	,658	,612	,341	,777	,220	,314	,424	,395	,673	,583	,355	,287	,183	,234	,125	,557	,660	,616	,342	,316	,241	,778	,563	,345	,452	,397	,403	,207	,674	,558	,785	,432	,357	,187	,236	,664	,624	,587	,780	,705	,126	,242	,565	,398	,346	,456	,358	,405	,303	,569	,244	,595	,189	,566	,676	,361	,706	,589	,215	,786	,647	,348	,419	,406	,464	,680	,801	,362	,590	,409	,570	,788	,597	,572	,219	,311	,708	,598	,601	,651	,421	,792	,802	,611	,602	,410	,231	,688	,653	,248	,369	,190,364	,654	,659	,335	,480	,315	,221	,370	,613	,422	,425	,451	,614	,543	,235	,412	,343	,372	,775	,317	,222	,426	,453	,237	,559	,833	,804	,712	,834	,661	,808	,779	,617	,604	,433	,720	,816	,836	,347	,897	,243	,662	,454	,318	,675	,618	,898	,781	,376	,428	,665	,736	,567	,840	,625	,238	,359	,457	,399	,787	,591	,678	,434	,677	,349	,245	,458	,666	,620	,363	,127	,191	,782	,407	,436	,626	,571	,465	,681	,246	,707	,350	,599	,668	,790	,460	,249	,682	,573	,411	,803	,789	,709	,365	,440	,628	,689	,374	,423	,466	,793	,250	,371	,481	,574	,413	,603	,366	,468	,655	,900	,805	,615	,684	,710	,429	,794	,252	,373	,605	,848	,690	,713	,632	,482	,806	,427	,904,414	,223	,663	,692	,835	,619	,472	,455	,796	,809	,714	,721	,837	,716	,864	,810	,606	,912	,722	,696	,377	,435	,817	,319	,621	,812	,484	,430	,838	,667	,488	,239	,378	,459	,622	,627	,437	,380	,818	,461	,496	,669	,679	,724	,841	,629	,351	,467	,438	,737	,251	,462	,442	,441	,469	,247	,683	,842	,738	,899	,670	,783	,849	,820	,728	,928	,791	,367	,901	,630	,685	,844	,633	,711	,253	,691	,824	,902	,686	,740	,850	,375	,444	,470	,483	,415	,485	,905	,795	,473	,634	,744	,852	,960	,865	,693	,797	,906	,715	,807	,474	,636	,694	,254	,717	,575	,913	,798	,811	,379	,697	,431	,607	,489	,866	,723	,486	,908	,718	,813	,476	,856	,839	,725	,698	,914	,752	,868,819	,814	,439	,929	,490	,623	,671	,739	,916	,463	,843	,381	,497	,930	,821	,726	,961	,872	,492	,631	,729	,700	,443	,741	,845	,920	,382	,822	,851	,730	,498	,880	,742	,445	,471	,635	,932	,687	,903	,825	,500	,846	,745	,826	,732	,446	,962	,936	,475	,853	,867	,637	,907	,487	,695	,746	,828	,753	,854	,857	,504	,799	,255	,964	,909	,719	,477	,915	,638	,748	,944	,869	,491	,699	,754	,858	,478	,968	,383	,910	,815	,976	,870	,917	,727	,493	,873	,701	,931	,756	,860	,499	,731	,823	,922	,874	,918	,502	,933	,743	,760	,881	,494	,702	,921	,501	,876	,847	,992	,447	,733	,827	,934	,882	,937	,963	,747	,505	,855	,924	,734	,829	,965	,938	,884	,506	,749	,945,966	,755	,859	,940	,830	,911	,871	,639	,888	,479	,946	,750	,969	,508	,861	,757	,970	,919	,875	,862	,758	,948	,977	,923	,972	,761	,877	,952	,495	,703	,935	,978	,883	,762	,503	,925	,878	,735	,993	,885	,939	,994	,980	,926	,764	,941	,967	,886	,831	,947	,507	,889	,984	,751	,942	,996	,971	,890	,509	,949	,973	,1000	,892	,950	,863	,759	,1008	,510	,979	,953	,763	,974	,954	,879	,981	,982	,927	,995	,765	,956	,887	,985	,997	,986	,943	,891	,998	,766	,511	,988	,1001	,951	,1002	,893	,975	,894	,1009	,955	,1004	,1010	,957	,983	,958	,987	,1012	,999	,1016	,767	,989	,1003	,990	,1005	,959	,1011	,1013	,895	,1006	,1014	,1017	,1018	,991	,1020	,1007	,1015	,1019	,1021	,1022	,1023])
        elif self.N == 512:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,64	,9	,6	,17	,10	,18	,128	,12	,33	,65	,20	,256	,34	,24	,36	,7	,129	,66	,11	,40	,68	,130	,19	,13	,48	,14	,72	,257	,21	,132	,35	,258	,26	,80	,37	,25	,22	,136	,260	,264	,38	,96	,67	,41	,144	,28	,69	,42	,49	,74	,272	,160	,288	,192	,70	,44	,131	,81	,50	,73	,15	,320	,133	,52	,23	,134	,384	,76	,137	,82	,56	,27	,97	,39	,259	,84	,138	,145	,261	,29	,43	,98	,88	,140	,30	,146	,71	,262	,265	,161	,45	,100	,51	,148	,46	,75	,266	,273	,104	,162	,53	,193	,152	,77	,164	,268	,274	,54	,83	,57	,112	,135	,78	,289	,194	,85	,276	,58	,168,139	,99	,86	,60	,280	,89	,290	,196	,141	,101	,147	,176	,142	,321	,31	,200	,90	,292	,322	,263	,149	,102	,105	,304	,296	,163	,92	,47	,267	,385	,324	,208	,386	,150	,153	,165	,106	,55	,328	,113	,154	,79	,269	,108	,224	,166	,195	,270	,275	,291	,59	,169	,114	,277	,156	,87	,197	,116	,170	,61	,281	,278	,177	,293	,388	,91	,198	,172	,120	,201	,336	,62	,282	,143	,103	,178	,294	,93	,202	,323	,392	,297	,107	,180	,151	,209	,284	,94	,204	,298	,400	,352	,325	,155	,210	,305	,300	,109	,184	,115	,167	,225	,326	,306	,157	,329	,110	,117	,212	,171	,330	,226	,387	,308	,216	,416	,271	,279	,158	,337	,118	,332	,389	,173	,121	,199	,179	,228,338	,312	,390	,174	,393	,283	,122	,448	,353	,203	,63	,340	,394	,181	,295	,285	,232	,124	,205	,182	,286	,299	,354	,211	,401	,185	,396	,344	,240	,206	,95	,327	,402	,356	,307	,301	,417	,213	,186	,404	,227	,418	,302	,360	,111	,331	,214	,309	,188	,449	,217	,408	,229	,159	,420	,310	,333	,119	,339	,218	,368	,230	,391	,313	,450	,334	,233	,175	,123	,341	,220	,314	,424	,395	,355	,287	,183	,234	,125	,342	,316	,241	,345	,452	,397	,403	,207	,432	,357	,187	,236	,126	,242	,398	,346	,456	,358	,405	,303	,244	,189	,361	,215	,348	,419	,406	,464	,362	,409	,219	,311	,421	,410	,231	,248	,369	,190	,364	,335	,480	,315	,221	,370	,422	,425	,451	,235	,412,343	,372	,317	,222	,426	,453	,237	,433	,347	,243	,454	,318	,376	,428	,238	,359	,457	,399	,434	,349	,245	,458	,363	,127	,191	,407	,436	,465	,246	,350	,460	,249	,411	,365	,440	,374	,423	,466	,250	,371	,481	,413	,366	,468	,429	,252	,373	,482	,427	,414	,223	,472	,455	,377	,435	,319	,484	,430	,488	,239	,378	,459	,437	,380	,461	,496	,351	,467	,438	,251	,462	,442	,441	,469	,247	,367	,253	,375	,444	,470	,483	,415	,485	,473	,474	,254	,379	,431	,489	,486	,476	,439	,490	,463	,381	,497	,492	,443	,382	,498	,445	,471	,500	,446	,475	,487	,504	,255	,477	,491	,478	,383	,493	,499	,502	,494	,501	,447	,505	,506	,479	,508	,495	,503	,507	,509	,510	,511])
        elif self.N == 256:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,64	,9	,6	,17	,10	,18	,128	,12	,33	,65	,20	,34	,24	,36	,7	,129	,66	,11	,40	,68	,130	,19	,13	,48	,14	,72	,21	,132	,35	,26	,80	,37	,25	,22	,136	,38	,96	,67	,41	,144	,28	,69	,42	,49	,74	,160	,192	,70	,44	,131	,81	,50	,73	,15	,133	,52	,23	,134	,76	,137	,82	,56	,27	,97	,39	,84	,138	,145	,29	,43	,98	,88	,140	,30	,146	,71	,161	,45	,100	,51	,148	,46	,75	,104	,162	,53	,193	,152	,77	,164	,54	,83	,57	,112	,135	,78	,194	,85	,58	,168	,139	,99	,86	,60	,89	,196	,141	,101	,147	,176	,142	,31	,200	,90	,149	,102	,105	,163	,92,47	,208	,150	,153	,165	,106	,55	,113	,154	,79	,108	,224	,166	,195	,59	,169	,114	,156	,87	,197	,116	,170	,61	,177	,91	,198	,172	,120	,201	,62	,143	,103	,178	,93	,202	,107	,180	,151	,209	,94	,204	,155	,210	,109	,184	,115	,167	,225	,157	,110	,117	,212	,171	,226	,216	,158	,118	,173	,121	,199	,179	,228	,174	,122	,203	,63	,181	,232	,124	,205	,182	,211	,185	,240	,206	,95	,213	,186	,227	,111	,214	,188	,217	,229	,159	,119	,218	,230	,233	,175	,123	,220	,183	,234	,125	,241	,207	,187	,236	,126	,242	,244	,189	,215	,219	,231	,248	,190	,221	,235	,222	,237	,243	,238	,245	,127	,191	,246	,249	,250	,252	,223	,239	,251	,247	,253	,254	,255])
        elif self.N == 128:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,64	,9	,6	,17	,10	,18	,12	,33	,65	,20	,34	,24	,36	,7	,66	,11	,40	,68	,19	,13	,48	,14	,72	,21	,35	,26	,80	,37	,25	,22	,38	,96	,67	,41	,28	,69	,42	,49	,74	,70	,44	,81	,50	,73	,15	,52	,23	,76	,82	,56	,27	,97	,39	,84	,29	,43	,98	,88	,30	,71	,45	,100	,51	,46	,75	,104	,53	,77	,54	,83	,57	,112	,78	,85	,58	,99	,86	,60	,89	,101	,31	,90	,102	,105	,92	,47	,106	,55	,113	,79	,108	,59	,114	,87	,116	,61	,91	,120	,62	,103	,93	,107	,94	,109	,115	,110	,117	,118	,121	,122	,63	,124	,95	,111	,119	,123	,125	,126	,127])
        elif self.N == 64:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,9	,6	,17	,10	,18	,12	,33	,20	,34	,24	,36	,7	,11	,40	,19	,13	,48	,14	,21	,35	,26	,37	,25	,22	,38	,41	,28	,42	,49	,44	,50	,15	,52	,23	,56	,27	,39	,29	,43	,30	,45	,51	,46	,53	,54	,57	,58	,60	,31	,47	,55	,59	,61	,62	,63])
        elif self.N == 32:
            pw_order_array = np.array([0	,1	,2	,4	,8	,16	,3	,5	,9	,6	,17	,10	,18	,12	,20	,24	,7	,11	,19	,13	,14	,21	,26	,25	,22	,28	,15	,23	,27	,29	,30	,31])
        elif self.N == 16:
            pw_order_array = np.array([0	,1	,2	,4	,8	,3	,5	,9	,6	,10	,12	,7	,11	,13	,14	,15])
        elif self.N == 8:
            pw_order_array = np.array([0	,1	,2	,4	,3	,5	,6	,7])
        elif self.N == 4:
            pw_order_array = np.array([0	,1	,2	,3])
        elif self.N == 2:
            pw_order_array = np.array([0	,1])
        else:
            self.pw_order_array = np.array([0	,1	,2	,4	,8	,16	,32	,3	,5	,64	,9	,6	,17	,10	,18	,128	,12	,33	,65	,20	,256	,34	,24	,36	,7	,129	,66	,512	,11	,40	,68	,130	,19	,13	,48	,14	,72	,257	,21	,132	,35	,258	,26	,513	,80	,37	,25	,22	,136	,260	,264	,38	,514	,96	,67	,41	,144	,28	,69	,42	,516	,49	,74	,272	,160	,520	,288	,528	,192	,544	,70	,44	,131	,81	,50	,73	,15	,320	,133	,52	,23	,134	,384	,76	,137	,82	,56	,27	,97	,39	,259	,84	,138	,145	,261	,29	,43	,98	,515	,88	,140	,30	,146	,71	,262	,265	,161	,576	,45	,100	,640	,51	,148	,46	,75	,266	,273	,517	,104	,162	,53	,193	,152	,77	,164	,768	,268	,274,518	,54	,83	,57	,521	,112	,135	,78	,289	,194	,85	,276	,522	,58	,168	,139	,99	,86	,60	,280	,89	,290	,529	,524	,196	,141	,101	,147	,176	,142	,530	,321	,31	,200	,90	,545	,292	,322	,532	,263	,149	,102	,105	,304	,296	,163	,92	,47	,267	,385	,546	,324	,208	,386	,150	,153	,165	,106	,55	,328	,536	,577	,548	,113	,154	,79	,269	,108	,578	,224	,166	,519	,552	,195	,270	,641	,523	,275	,580	,291	,59	,169	,560	,114	,277	,156	,87	,197	,116	,170	,61	,531	,525	,642	,281	,278	,526	,177	,293	,388	,91	,584	,769	,198	,172	,120	,201	,336	,62	,282	,143	,103	,178	,294	,93	,644	,202	,592	,323	,392	,297	,770	,107	,180	,151	,209	,284	,648,94	,204	,298	,400	,608	,352	,325	,533	,155	,210	,305	,547	,300	,109	,184	,534	,537	,115	,167	,225	,326	,306	,772	,157	,656	,329	,110	,117	,212	,171	,776	,330	,226	,549	,538	,387	,308	,216	,416	,271	,279	,158	,337	,550	,672	,118	,332	,579	,540	,389	,173	,121	,553	,199	,784	,179	,228	,338	,312	,704	,390	,174	,554	,581	,393	,283	,122	,448	,353	,561	,203	,63	,340	,394	,527	,582	,556	,181	,295	,285	,232	,124	,205	,182	,643	,562	,286	,585	,299	,354	,211	,401	,185	,396	,344	,586	,645	,593	,535	,240	,206	,95	,327	,564	,800	,402	,356	,307	,301	,417	,213	,568	,832	,588	,186	,646	,404	,227	,896	,594	,418	,302	,649	,771	,360	,539	,111	,331,214	,309	,188	,449	,217	,408	,609	,596	,551	,650	,229	,159	,420	,310	,541	,773	,610	,657	,333	,119	,600	,339	,218	,368	,652	,230	,391	,313	,450	,542	,334	,233	,555	,774	,175	,123	,658	,612	,341	,777	,220	,314	,424	,395	,673	,583	,355	,287	,183	,234	,125	,557	,660	,616	,342	,316	,241	,778	,563	,345	,452	,397	,403	,207	,674	,558	,785	,432	,357	,187	,236	,664	,624	,587	,780	,705	,126	,242	,565	,398	,346	,456	,358	,405	,303	,569	,244	,595	,189	,566	,676	,361	,706	,589	,215	,786	,647	,348	,419	,406	,464	,680	,801	,362	,590	,409	,570	,788	,597	,572	,219	,311	,708	,598	,601	,651	,421	,792	,802	,611	,602	,410	,231	,688	,653	,248	,369	,190,364	,654	,659	,335	,480	,315	,221	,370	,613	,422	,425	,451	,614	,543	,235	,412	,343	,372	,775	,317	,222	,426	,453	,237	,559	,833	,804	,712	,834	,661	,808	,779	,617	,604	,433	,720	,816	,836	,347	,897	,243	,662	,454	,318	,675	,618	,898	,781	,376	,428	,665	,736	,567	,840	,625	,238	,359	,457	,399	,787	,591	,678	,434	,677	,349	,245	,458	,666	,620	,363	,127	,191	,782	,407	,436	,626	,571	,465	,681	,246	,707	,350	,599	,668	,790	,460	,249	,682	,573	,411	,803	,789	,709	,365	,440	,628	,689	,374	,423	,466	,793	,250	,371	,481	,574	,413	,603	,366	,468	,655	,900	,805	,615	,684	,710	,429	,794	,252	,373	,605	,848	,690	,713	,632	,482	,806	,427	,904,414	,223	,663	,692	,835	,619	,472	,455	,796	,809	,714	,721	,837	,716	,864	,810	,606	,912	,722	,696	,377	,435	,817	,319	,621	,812	,484	,430	,838	,667	,488	,239	,378	,459	,622	,627	,437	,380	,818	,461	,496	,669	,679	,724	,841	,629	,351	,467	,438	,737	,251	,462	,442	,441	,469	,247	,683	,842	,738	,899	,670	,783	,849	,820	,728	,928	,791	,367	,901	,630	,685	,844	,633	,711	,253	,691	,824	,902	,686	,740	,850	,375	,444	,470	,483	,415	,485	,905	,795	,473	,634	,744	,852	,960	,865	,693	,797	,906	,715	,807	,474	,636	,694	,254	,717	,575	,913	,798	,811	,379	,697	,431	,607	,489	,866	,723	,486	,908	,718	,813	,476	,856	,839	,725	,698	,914	,752	,868,819	,814	,439	,929	,490	,623	,671	,739	,916	,463	,843	,381	,497	,930	,821	,726	,961	,872	,492	,631	,729	,700	,443	,741	,845	,920	,382	,822	,851	,730	,498	,880	,742	,445	,471	,635	,932	,687	,903	,825	,500	,846	,745	,826	,732	,446	,962	,936	,475	,853	,867	,637	,907	,487	,695	,746	,828	,753	,854	,857	,504	,799	,255	,964	,909	,719	,477	,915	,638	,748	,944	,869	,491	,699	,754	,858	,478	,968	,383	,910	,815	,976	,870	,917	,727	,493	,873	,701	,931	,756	,860	,499	,731	,823	,922	,874	,918	,502	,933	,743	,760	,881	,494	,702	,921	,501	,876	,847	,992	,447	,733	,827	,934	,882	,937	,963	,747	,505	,855	,924	,734	,829	,965	,938	,884	,506	,749	,945,966	,755	,859	,940	,830	,911	,871	,639	,888	,479	,946	,750	,969	,508	,861	,757	,970	,919	,875	,862	,758	,948	,977	,923	,972	,761	,877	,952	,495	,703	,935	,978	,883	,762	,503	,925	,878	,735	,993	,885	,939	,994	,980	,926	,764	,941	,967	,886	,831	,947	,507	,889	,984	,751	,942	,996	,971	,890	,509	,949	,973	,1000	,892	,950	,863	,759	,1008	,510	,979	,953	,763	,974	,954	,879	,981	,982	,927	,995	,765	,956	,887	,985	,997	,986	,943	,891	,998	,766	,511	,988	,1001	,951	,1002	,893	,975	,894	,1009	,955	,1004	,1010	,957	,983	,958	,987	,1012	,999	,1016	,767	,989	,1003	,990	,1005	,959	,1011	,1013	,895	,1006	,1014	,1017	,1018	,991	,1020	,1007	,1015	,1019	,1021	,1022	,1023])
            print("not supported by PW")
            exit()
        #self.info_sel = np.zeros(self.N, dtype=int)
        self.info_sel = []
        pw_sel_index = self.N - 1
        pwSortedOrder = np.zeros(self.N)
        for i in range(0, self.N):
            pwSortedOrder[i] = pw_order_array[pw_sel_index]
            pw_sel_index -= 1

        for i in range(0, self.K):
            #self.info_sel[int(pwSortedOrder[i])] = 1
            self.info_sel.append(int(pwSortedOrder[i]))
        self.info_sel = np.asarray(self.info_sel)
        self.info_set = np.sort(self.info_sel)


    def Info_Bha(self):
        sort = np.copy(self.bha_list)
        sort = np.sort(sort) #sort bha_list (ascending order)
        threshold = sort[self.K - 1]
        self.frozen_set = []
        for i in range(self.N):
            if(self.bha_list[i]>threshold):
                self.frozen_set.append(i)
        self.info_set = [i for i in range(self.N) if i not in self.frozen_set]
        if(len(self.info_set)!=self.K):
            print("info size is wrong")

    def Generator_Polar(self):
        '''
        #permutation_matrix
        self.Permutation_mat = np.zeros((self.N,self.N))
        for i in range(self.N):
            i_reversal = int(bin(i)[2:].zfill(self.n)[::-1], 2)
            self.Permutation_mat[i, i_reversal] = 1
        '''
        #permutation_vector
        self.rev = [bit_reverse(i, self.n) for i in range(self.N)]
        #F_N
        F_2 = np.array([[1,0],
                        [1,1]])
        self.F_N = np.array([[1]])
        for i in range(self.n):
            self.F_N = np.kron(self.F_N, F_2)

        #self.G_Polar = np.matmul(self.Permutation_mat,self.F_N)
        self.G_Polar = self.F_N[self.rev, :]

    def crc_generator_matrix(self):
        #K is message length == N_message
        #K + self.CRC is the size of information sets ==N_information

        r = self.CRC
        m = self.M
        k = m+r

        if self.CRC == 4:
            self.CRC_poly = np.array([1,0,0,1,1], dtype=np.int8)
        elif self.CRC == 6:
            self.CRC_poly = np.array([1,1,0,0,0,0,1], dtype=np.int8)
        elif self.CRC == 8:
            self.CRC_poly = np.array([1,0,0,0,0,0,1,1,1], dtype=np.int8)
        elif self.CRC == 11:
            self.CRC_poly = np.array([1,1,1,0,0,0,1,0,0,0,0,1], dtype=np.int8)
        elif self.CRC == 16:
            self.CRC_poly = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1], dtype=np.int8)
            #self.CRC_poly = np.array([1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1], dtype=np.int8)
        elif self.CRC == 24:
            self.CRC_poly = np.array([1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1], dtype=np.int8)
        else:
            print("In SelCRCpoly(): CRC", self.CRC, " are not supported!!!")

        G = np.zeros((m, k), dtype=np.int8)

        for i in range(m):
            G[i, i] = 1
            # 這裡模擬訊息左移 r 位後做模2除法得到的餘數
            msg = np.zeros(m + r, dtype=np.int8)
            msg[i] = 1  # 單位訊號
            # 做mod2除法來模擬CRC餘數
            for j in range(i, i + r + 1):
                if msg[j] == 1:
                    for l in range(r + 1):
                        if j + l < m + r:
                            msg[j + l] ^= self.CRC_poly[l]
            G[i, m:] = msg[m:]  # 取餘數部分作為CRC bits
        self.G_CRC = G.copy()

    def compute_crc_syndrome(self, data_bits): ##, poly_bits):
        """
        data_bits: list of int or NumPy array (0/1 bits)
        poly_bits: list of int or NumPy array (CRC polynomial bits)
        return NumPy array representing the CRC syndrome
        """
        data = np.array(data_bits).copy()
        poly = self.CRC_poly.copy()

        data = data.astype(np.int8)
        poly = poly.astype(np.int8)

        poly_len = len(poly)

        for i in range(len(data) - poly_len + 1):
            if data[i] == 1:
                data[i:i + poly_len] ^= poly

        return data[-(poly_len - 1):]

    def book(self):
        #level_book & reverse_book construction
        self.level_book = []    # 1 + num of consecutive 0 from MSB
        self.reverse_book = []
        for dec_bit in range(self.N):
            dec_bit_r = int(bin(dec_bit)[2:].zfill(self.n)[::-1], 2)
            tmp = bin(dec_bit_r)[2:].zfill(self.n)
            level = 1
            for j in range(self.n):
                if(tmp[j] == '0'):
                    level = level + 1
                else:
                    break
            self.level_book.append(level)
            self.reverse_book.append(dec_bit_r)

        self.level_book[0] = self.n #bit 0 condition
        #print(self.level_book)
        #print(self.reverse_book)

    def batch_generator(self, std_noise):#return (Batch,N) &(Batch,M)
        ##random message with BPSK done
        ##include frozen set

        ##message = np.random.randint(0, 2, size= self.M).astype(np.int8)
        ##message = np.zeros(self.M, dtype=np.int8)
        message = np.ones(self.M, dtype=np.int8)
        if(self.CRC>0):
            information = np.matmul(message, self.G_CRC) %2  #(M,)(M, K)->(K,)
        else:
            information = message
        codeword = np.zeros(self.N, dtype=np.int8)
        ##for i, val in enumerate(self.info_set):
        ##    codeword[val] = information[i]
        codeword[self.info_set]= information
        codeword_rev = codeword[self.rev]
        codeword_polar = np.matmul(codeword, self.G_Polar) %2
        ##debug print
        '''
        print(f'message is {message}')
        print(f'info set [{self.info_set}]')
        print(f'codeword is {codeword}')
        print(f'codeword_rev is {codeword_rev}')
        print(f'codeword_polar is {codeword_polar}')
        '''
        codeword_bpsk = 1 - (2*codeword_polar)

        noise = np.random.normal(scale= std_noise, size= codeword_bpsk.shape)
        ##noise = 0
        x = codeword_bpsk + noise
        llr = 2*x / (std_noise**2)
        y = message
        ##return llr, codeword_rev
        return llr, y
    
    def scl_decoder(self, llr, flip_arr):

        # init trees, root at level 0
        llr_tree = np.zeros((self.n+1, self.N), dtype =np.float64)
        bit_tree = np.zeros((self.n+1, self.N), dtype =np.int8)

        # LLR receive at root node
        llr_tree[0] = llr.copy()
        pm_log = []
        
        ##debug print
        ##print('begin')
        
        # recursive function
        def recurse(level, idx_node, paths, idx_bit):
            ##debug print
            '''
            for _ in range(level):
                print('    ',end=' ')
            print(f'r[{level},{idx_node}]')
            '''
            # leaf
            if level == self.n:
                ##debug print
                '''
                for j in range(level):
                    print('    ',end=' ') 
                print(f'leaf[{level},{idx_node}]')
                '''
                new_paths = []
                for path in paths:
                    LLR = path.llr_tree[level][idx_bit]
                    ##debug print
                    ##print(f'llr at (level={level},idx_bit={idx_bit})={LLR}')
                    if idx_node in self.info_set:
                        for u in [0, 1]:
                            p = path.copy()
                            p.bit_tree[level][idx_bit] = u
                            if(u==0 and LLR<0) or (u==1 and LLR>=0): #path penalty
                                p.pm+=abs(LLR)
                            else:
                                p.pm+=0
                            new_paths.append(p)
                    else:
                        p = path.copy()
                        p.bit_tree[level][idx_bit] = 0
                        if(LLR<0): #path penalty
                            p.pm+=abs(LLR)
                        else:
                            p.pm+=0
                        new_paths.append(p)
                ##debug print
                '''        
                print(f'len of new_paths:{len(new_paths)}')
                for path in new_paths:
                    print(f'u={path.bit_tree[level][idx_bit]},pm={path.pm}')
                '''
                #keep path metric at all idx for ML
                metrics = [p.pm for p in new_paths]
                pad_len = 2*self.List - len(metrics)
                if pad_len > 0:
                    metrics.extend([PAD] * pad_len)
                pm_log.append(metrics)
                #prune the list
                new_paths.sort(key=lambda x: x.pm)
                ##debug print
                '''
                print(f'idx_bit={idx_bit}, after sort')
                for path in new_paths:
                    print(f'u={path.bit_tree[level][idx_bit]},pm={path.pm}')
                '''
                if idx_bit in flip_arr:
                    return new_paths[self.List:]
                else:
                    return new_paths[:self.List]

            # f stage
            for path in paths:
                for i in range(2**(self.n-level-1)):
                    d = 2**level
                    L = path.llr_tree[level][idx_bit+(2*i)*d]
                    R = path.llr_tree[level][idx_bit+(2*i+1)*d]
                    path.llr_tree[level+1][idx_bit+(2*i)*d] = f_func(L, R)
                    ##debug print
                    '''
                    for _ in range(level):
                        print('    ',end='')                    
                    print(f'f[{level},{idx_bit+(2*i)*d},{idx_bit+(2*i+1)*d}]')
                    for j in range(self.n+1):
                        for _ in range(level):
                            print('    ',end='')  
                        print(path.llr_tree[j])
                    input("press enter to conti")
                    '''
            paths = recurse(level+1, 2*idx_node, paths, idx_bit)
            # g stage
            for path in paths:
                for i in range(2**(self.n-level-1)):
                    d = 2**level
                    L = path.llr_tree[level][idx_bit+(2*i)*d]
                    R = path.llr_tree[level][idx_bit+(2*i+1)*d]
                    uL = path.bit_tree[level+1][idx_bit+(2*i)*d]
                    path.llr_tree[level+1][idx_bit+(2*i+1)*d] = g_func(L, R, uL)
                    ##debug print
                    '''
                    for _ in range(level):
                        print('    ',end='')                    
                    print(f'g[{level},{idx_bit+(2*i)*d},{idx_bit+(2*i+1)*d}]')
                    for j in range(self.n+1):
                        for _ in range(level):
                            print('    ',end='')
                        print(path.llr_tree[j])
                    input("press enter to conti")
                    '''
            paths = recurse(level+1, 2*idx_node+1, paths, idx_bit+(2**level))
            # partial sum
            ##if level > 0:
            for path in paths:
                for i in range(2**(self.n-level-1)):
                    d = 2**level
                    uL = path.bit_tree[level+1][idx_bit+(2*i)*d]
                    uR = path.bit_tree[level+1][idx_bit+(2*i+1)*d]
                    path.bit_tree[level][idx_bit+(2*i)*d] = uL ^ uR
                    path.bit_tree[level][idx_bit+(2*i+1)*d] = uR
                    ##debug print
                    '''
                    for _ in range(level):
                        print('    ',end='')
                    print(f'ps[{level},{idx_bit+(2*i)*d},{idx_bit+(2*i+1)*d}]')
                    for j in range(self.n+1):
                        for _ in range(level):
                            print('    ',end='')
                        print(path.bit_tree[j])
                    input("press enter to conti")
                    '''
            return paths
        
        # start with one path
        init_path = Path(llr_tree, bit_tree, 0.0)
        paths = [init_path]
        paths = recurse(0, 0, paths, 0) # start recursion from root level
        
        ##debug print
        ##print('end')
        
        # select path with least pm and crc passed
        rev = np.array([bit_reverse(i, self.n) for i in range(self.N)])
        if(self.CRC>0):
            crc_paths=[]
            for path in paths:
                u_rev = path.bit_tree[self.n]
                u = r_rev[rev]
                info = u[self.info_set]
                if(np.any(compute_crc_syndrome(info)) == 0):
                    crc_paths.append(path)
            best = min(crc_paths, key=lambda x: x.pm)
            # extract message
            u_rev = best.bit_tree[self.n]
            u= u_rev[rev]
            info = u[self.info_set]
            y = info[:self.M]
        else:
            # select best
            best = min(paths, key=lambda x: x.pm)
            # extract message
            u_rev = best.bit_tree[self.n]
            u= u_rev[rev]
            info = u[self.info_set]
            y = info[:self.M]
        ##debug print
        '''
        print('decoder output before slicing but after reversing',u_rev)
        print('decoder output before slicing but after reversing',u)
        print('decoder output',y)
        '''
        
        ##return u_rev, pm_log
        return y, pm_log

    def Flip_choice(self, PM_in):#PM(Batch, 2*List, self.N) should input only one batch -> PM_in(2*List, N)
        #flip_array = np.zeros(self.N, dtype=np.uint8) # 1 for index needed flip, 0 for others
        n_l = int(np.log2(self.List))
        flip_metric = np.zeros(len(self.info_set))
        flip_indice = []
        '''
        #compute flip metric in information set/first log_2(N) bit panyihan method
        for i, idx in enumerate(self.info_set):
            numerator = 0
            denominator = 0
            for j in range(self.List):
                numerator += np.exp(-PM_in[j, idx])
                denominator += np.exp(-PM_in[j+self.List, idx])
            flip_metric[i] = numerator/pow(denominator, self.Alpha)
            '''
        #compute flip metric in information set/first log_2(N) bit by new method
        for i, idx in enumerate(self.info_set):
            survive = 0
            discard = 0
            for j in range(self.List):
                survive += PM_in[j, idx]
                discard += PM_in[j+self.List, idx]
            flip_metric[i] = survive - discard

        #point out the T significant candidates info bit and transform them into polar code index
        order = np.argsort(flip_metric[n_l:])[:self.T_flip]
        flip_indice = [ self.info_set[i+n_l] for i in order ]

        #flip_array[flip_indice] = 1  #retrun flip candidate in boolyn form length N vector if needed

        return flip_indice


# Example usage:
# u_hat = scl_decode(llr, info_set, L=4)
