"""
文本分块工具
"""
from typing import List, Optional
import re
import os
from loguru import logger

class TextSplitter:
    """文本分块工具类"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n",
        keep_separator: bool = False,
        use_model: bool = False,
        model_path: str = "damo/nlp_bert_document-segmentation_chinese-base",
        device: str = "cuda"
    ):
        """
        初始化
        
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            separator: 分隔符，用于初步分割文本
            keep_separator: 是否在分块结果中保留分隔符
            use_model: 是否使用BERT模型进行语义分句
            model_path: BERT模型路径
            device: 设备类型，'cuda'或'cpu'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
        self.use_model = use_model
        self.model_path = model_path
        self.device = device
        
        # 如果使用模型，则加载模型
        if self.use_model:
            try:
                from modelscope.pipelines import pipeline
                self.sent_split_pp = pipeline(
                    task="document-segmentation",
                    model=model_path,
                    device=device
                )
                logger.info(f"成功加载文档分割模型: {model_path}")
            except ImportError:
                logger.warning("未安装modelscope，将使用规则分句")
                self.use_model = False
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                self.use_model = False
        
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        # 如果文本长度小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        # 使用模型进行分句
        if self.use_model:
            return self._split_text_with_model(text)
        else:
            return self._split_text_with_rules(text)
    
    def _split_text_with_model(self, text: str) -> List[str]:
        """使用BERT模型进行语义分句"""
        try:
            result = self.sent_split_pp(documents=text)
            sent_list = [i for i in result["text"].split("\n\t") if i]
            
            # 处理过长的句子
            final_chunks = []
            for sent in sent_list:
                if len(sent) <= self.chunk_size:
                    final_chunks.append(sent)
                else:
                    # 对过长的句子进行再分割
                    sub_chunks = self._split_text_with_rules(sent)
                    final_chunks.extend(sub_chunks)
            
            return final_chunks
        except Exception as e:
            logger.error(f"模型分句失败: {e}，回退到规则分句")
            return self._split_text_with_rules(text)
    
    def _split_text_with_rules(self, text: str) -> List[str]:
        """使用规则进行分句"""
        # 先按标点符号进行初步分句
        text = re.sub(r'([;；.!?。！？\?])([^"''])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"''"」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"''"」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["''"」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        sent_list = [i for i in text.split("\n") if i]
        
        # 处理过长的句子
        chunks = []
        for ele in sent_list:
            if len(ele) <= self.chunk_size:
                chunks.append(ele)
            else:
                # 对过长的句子进行再分割
                ele1 = re.sub(r'([,，.]["''"」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) <= self.chunk_size:
                        chunks.append(ele_ele1)
                    else:
                        # 继续分割
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["''"」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        chunks.extend([i for i in ele2_ls if i])
        
        # 合并短句子
        final_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks
        
    def split_documents(self, documents: List[str]) -> List[str]:
        """
        分割多个文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文档的文本块列表
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc)
            chunks.extend(doc_chunks)
        return chunks 