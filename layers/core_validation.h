/* Copyright (c) 2015-2019 The Khronos Group Inc.
 * Copyright (c) 2015-2019 Valve Corporation
 * Copyright (c) 2015-2019 LunarG, Inc.
 * Copyright (C) 2015-2019 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Courtney Goeltzenleuchter <courtneygo@google.com>
 * Author: Tobin Ehlis <tobine@google.com>
 * Author: Chris Forbes <chrisf@ijw.co.nz>
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Dave Houlton <daveh@lunarg.com>
 */

#pragma once
#include "core_validation_error_enums.h"
#include "core_validation_types.h"
#include "descriptor_sets.h"
#include "shader_validation.h"
#include "gpu_validation.h"
#include "vk_layer_logging.h"
#include "vulkan/vk_layer.h"
#include "vk_typemap_helper.h"
#include "vk_layer_data.h"
#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <deque>

enum SyncScope {
    kSyncScopeInternal,
    kSyncScopeExternalTemporary,
    kSyncScopeExternalPermanent,
};

enum FENCE_STATUS { FENCE_UNSIGNALED, FENCE_INFLIGHT, FENCE_RETIRED };

class FENCE_STATE {
   public:
    VkFence fence;
    VkFenceCreateInfo createInfo;
    std::pair<VkQueue, uint64_t> signaler;
    FENCE_STATUS state;
    SyncScope scope;

    // Default constructor
    FENCE_STATE() : state(FENCE_UNSIGNALED), scope(kSyncScopeInternal) {}
};

class SEMAPHORE_STATE : public BASE_NODE {
   public:
    std::pair<VkQueue, uint64_t> signaler;
    bool signaled;
    SyncScope scope;
};

class EVENT_STATE : public BASE_NODE {
   public:
    int write_in_use;
    bool needsSignaled;
    VkPipelineStageFlags stageMask;
};

class QUEUE_STATE {
   public:
    VkQueue queue;
    uint32_t queueFamilyIndex;
    std::unordered_map<VkEvent, VkPipelineStageFlags> eventToStageMap;
    std::unordered_map<QueryObject, bool> queryToStateMap;  // 0 is unavailable, 1 is available

    uint64_t seq;
    std::deque<CB_SUBMISSION> submissions;
};

class QUERY_POOL_STATE : public BASE_NODE {
   public:
    VkQueryPoolCreateInfo createInfo;
};

struct PHYSICAL_DEVICE_STATE {
    safe_VkPhysicalDeviceFeatures2 features2 = {};
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    uint32_t queue_family_count = 0;
    std::vector<VkQueueFamilyProperties> queue_family_properties;
    VkSurfaceCapabilitiesKHR surfaceCapabilities = {};
    std::vector<VkPresentModeKHR> present_modes;
    std::vector<VkSurfaceFormatKHR> surface_formats;
    uint32_t display_plane_property_count = 0;
};

// This structure is used to save data across the CreateGraphicsPipelines down-chain API call
struct create_graphics_pipeline_api_state {
    std::vector<safe_VkGraphicsPipelineCreateInfo> gpu_create_infos;
    std::vector<std::unique_ptr<PIPELINE_STATE>> pipe_state;
    const VkGraphicsPipelineCreateInfo* pCreateInfos;
};

// This structure is used modify parameters for the CreatePipelineLayout down-chain API call
struct create_pipeline_layout_api_state {
    std::vector<VkDescriptorSetLayout> new_layouts;
    VkPipelineLayoutCreateInfo modified_create_info;
};

// This structure is used modify and pass parameters for the CreateShaderModule down-chain API call

struct create_shader_module_api_state {
    uint32_t unique_shader_id;
    VkShaderModuleCreateInfo instrumented_create_info;
    std::vector<unsigned int> instrumented_pgm;
};

struct GpuQueue {
    VkPhysicalDevice gpu;
    uint32_t queue_family_index;
};

struct SubresourceRangeErrorCodes {
    const char *base_mip_err, *mip_count_err, *base_layer_err, *layer_count_err;
};

inline bool operator==(GpuQueue const& lhs, GpuQueue const& rhs) {
    return (lhs.gpu == rhs.gpu && lhs.queue_family_index == rhs.queue_family_index);
}

namespace std {
template <>
struct hash<GpuQueue> {
    size_t operator()(GpuQueue gq) const throw() {
        return hash<uint64_t>()((uint64_t)(gq.gpu)) ^ hash<uint32_t>()(gq.queue_family_index);
    }
};
}  // namespace std

using std::unordered_map;
struct GpuValidationState;

class CoreChecks : public ValidationObject {
   public:
    unordered_map<VkPipeline, std::unique_ptr<PIPELINE_STATE>> pipelineMap;
    unordered_map<VkFramebuffer, std::unique_ptr<FRAMEBUFFER_STATE>> frameBufferMap;
    unordered_map<VkShaderModule, std::unique_ptr<SHADER_MODULE_STATE>> shaderModuleMap;
    unordered_map<VkDescriptorUpdateTemplateKHR, std::unique_ptr<TEMPLATE_STATE>> desc_template_map;
    unordered_map<VkDescriptorPool, std::unique_ptr<DESCRIPTOR_POOL_STATE>> descriptorPoolMap;
    unordered_map<VkDescriptorSet, std::unique_ptr<cvdescriptorset::DescriptorSet>> setMap;
    unordered_map<VkCommandBuffer, std::unique_ptr<CMD_BUFFER_STATE>> commandBufferMap;
    unordered_map<VkCommandPool, std::unique_ptr<COMMAND_POOL_STATE>> commandPoolMap;
    unordered_map<VkPipelineLayout, std::unique_ptr<PIPELINE_LAYOUT_STATE>> pipelineLayoutMap;
    unordered_map<VkFence, std::unique_ptr<FENCE_STATE>> fenceMap;
    unordered_map<VkQueryPool, std::unique_ptr<QUERY_POOL_STATE>> queryPoolMap;
    unordered_map<VkSemaphore, std::unique_ptr<SEMAPHORE_STATE>> semaphoreMap;
    unordered_map<VkQueue, QUEUE_STATE> queueMap;
    unordered_map<VkEvent, EVENT_STATE> eventMap;

    unordered_map<VkRenderPass, std::shared_ptr<RENDER_PASS_STATE>> renderPassMap;
    unordered_map<VkDescriptorSetLayout, std::shared_ptr<cvdescriptorset::DescriptorSetLayout>> descriptorSetLayoutMap;

    std::unordered_set<VkQueue> queues;  // All queues under given device
    unordered_map<VkImage, std::vector<ImageSubresourcePair>> imageSubresourceMap;
    unordered_map<QueryObject, bool> queryToStateMap;
    unordered_map<VkSamplerYcbcrConversion, uint64_t> ycbcr_conversion_ahb_fmt_map;
    std::unordered_set<uint64_t> ahb_ext_formats_set;
    // Map for queue family index to queue count
    unordered_map<uint32_t, uint32_t> queue_family_index_map;

    // Used for instance versions of this object
    unordered_map<VkPhysicalDevice, PHYSICAL_DEVICE_STATE> physical_device_map;
    // Link to the device's physical-device data
    PHYSICAL_DEVICE_STATE* physical_device_state;

    // Link for derived device objects back to their parent instance object
    CoreChecks* instance_state;

    DeviceFeatures enabled_features = {};
    // Device specific data
    VkPhysicalDeviceMemoryProperties phys_dev_mem_props = {};
    VkPhysicalDeviceProperties phys_dev_props = {};
    // Device extension properties -- storing properties gathered from VkPhysicalDeviceProperties2KHR::pNext chain
    struct DeviceExtensionProperties {
        uint32_t max_push_descriptors;  // from VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors
        VkPhysicalDeviceDescriptorIndexingPropertiesEXT descriptor_indexing_props;
        VkPhysicalDeviceShadingRateImagePropertiesNV shading_rate_image_props;
        VkPhysicalDeviceMeshShaderPropertiesNV mesh_shader_props;
        VkPhysicalDeviceInlineUniformBlockPropertiesEXT inline_uniform_block_props;
        VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT vtx_attrib_divisor_props;
        VkPhysicalDeviceDepthStencilResolvePropertiesKHR depth_stencil_resolve_props;
        VkPhysicalDeviceCooperativeMatrixPropertiesNV cooperative_matrix_props;
        VkPhysicalDeviceTransformFeedbackPropertiesEXT transform_feedback_props;
    };
    DeviceExtensionProperties phys_dev_ext_props = {};
    std::vector<VkCooperativeMatrixPropertiesNV> cooperative_matrix_properties;
    bool external_sync_warning = false;
    std::unique_ptr<GpuValidationState> gpu_validation_state;
    uint32_t physical_device_count;

    // Class Declarations for helper functions
    cvdescriptorset::DescriptorSet* GetSetNode(VkDescriptorSet);
    DESCRIPTOR_POOL_STATE* GetDescriptorPoolState(const VkDescriptorPool);
    CMD_BUFFER_STATE* GetCBState(const VkCommandBuffer cb);
    PIPELINE_STATE* GetPipelineState(VkPipeline pipeline);
    RENDER_PASS_STATE* GetRenderPassState(VkRenderPass renderpass);
    std::shared_ptr<RENDER_PASS_STATE> GetRenderPassStateSharedPtr(VkRenderPass renderpass);
    FRAMEBUFFER_STATE* GetFramebufferState(VkFramebuffer framebuffer);
    COMMAND_POOL_STATE* GetCommandPoolState(VkCommandPool pool);
    SHADER_MODULE_STATE const* GetShaderModuleState(VkShaderModule module);
    FENCE_STATE* GetFenceState(VkFence fence);
    EVENT_STATE* GetEventState(VkEvent event);
    QUERY_POOL_STATE* GetQueryPoolState(VkQueryPool query_pool);
    QUEUE_STATE* GetQueueState(VkQueue queue);
    SEMAPHORE_STATE* GetSemaphoreState(VkSemaphore semaphore);
    PHYSICAL_DEVICE_STATE* GetPhysicalDeviceState(VkPhysicalDevice phys);
    PHYSICAL_DEVICE_STATE* GetPhysicalDeviceState();

    template <typename ExtProp>
    void GetPhysicalDeviceExtProperties(VkPhysicalDevice gpu, bool enabled, ExtProp* ext_prop) {
        assert(ext_prop);
        if (enabled) {
            *ext_prop = lvl_init_struct<ExtProp>();
            auto prop2 = lvl_init_struct<VkPhysicalDeviceProperties2KHR>(ext_prop);
            DispatchGetPhysicalDeviceProperties2KHR(gpu, &prop2);
        }
    }

    void PreCallRecordDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks *pAllocator);
    void RecordCreateRenderPassState(RenderPassCreateVersion rp_version, std::shared_ptr<RENDER_PASS_STATE> &render_pass,
        VkRenderPass *pRenderPass);
    void PostCallRecordCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass,
        VkResult result);
    void PostCallRecordCreateRenderPass2KHR(VkDevice device, const VkRenderPassCreateInfo2KHR *pCreateInfo,
        const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass, VkResult result);



    void UpdateStateCmdDrawDispatchType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point);
    void UpdateStateCmdDrawType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point);
    void PostCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
    void PostCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
    void PostCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
        uint32_t firstVertex, uint32_t firstInstance);
    void PostCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
        uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);
    void PostCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
        uint32_t stride);
    void PostCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
        uint32_t count, uint32_t stride);
    void PreCallRecordCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
        VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
        uint32_t stride);
    void PreCallRecordCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
        VkBuffer countBuffer, VkDeviceSize countBufferOffset,
        uint32_t maxDrawCount, uint32_t stride);
    void PreCallRecordCmdDrawMeshTasksNV(VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask);
    void PreCallRecordCmdDrawMeshTasksIndirectNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
        uint32_t drawCount, uint32_t stride);
    void PreCallRecordCmdDrawMeshTasksIndirectCountNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
        VkBuffer countBuffer, VkDeviceSize countBufferOffset,
        uint32_t maxDrawCount, uint32_t stride);





    bool PreCallValidateCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                                const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines,
                                                void* cgpl_state_data);
    bool VerifyQueueStateToSeq(QUEUE_STATE* initial_queue, uint64_t initial_seq);
    void ClearCmdBufAndMemReferences(CMD_BUFFER_STATE* cb_node);
    void ClearMemoryObjectBinding(uint64_t handle, VulkanObjectType type, VkDeviceMemory mem);
    void ResetCommandBufferState(const VkCommandBuffer cb);
    bool ValidateDeviceQueueFamily(uint32_t queue_family, const char* cmd_name, const char* parameter_name, const char* error_code,
                                   bool optional);
    bool ValidateBindBufferMemory(VkBuffer buffer, VkDeviceMemory mem, VkDeviceSize memoryOffset, const char* api_name);
    void RecordGetBufferMemoryRequirementsState(VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements);
    void UpdateBindBufferMemoryState(VkBuffer buffer, VkDeviceMemory mem, VkDeviceSize memoryOffset);
    PIPELINE_LAYOUT_STATE const* GetPipelineLayout(VkPipelineLayout pipeLayout);
    const TEMPLATE_STATE* GetDescriptorTemplateState(VkDescriptorUpdateTemplateKHR descriptor_update_template);
    bool ValidateGetImageMemoryRequirements2(const VkImageMemoryRequirementsInfo2* pInfo);
    void RecordGetImageMemoryRequiementsState(VkImage image, VkMemoryRequirements* pMemoryRequirements);
    void FreeCommandBufferStates(COMMAND_POOL_STATE* pool_state, const uint32_t command_buffer_count,
                                 const VkCommandBuffer* command_buffers);
    bool CheckCommandBuffersInFlight(COMMAND_POOL_STATE* pPool, const char* action, const char* error_code);
    bool CheckCommandBufferInFlight(const CMD_BUFFER_STATE* cb_node, const char* action, const char* error_code);
    bool VerifyQueueStateToFence(VkFence fence);
    void DecrementBoundResources(CMD_BUFFER_STATE const* cb_node);
    bool VerifyWaitFenceState(VkFence fence, const char* apiCall);
    void RetireFence(VkFence fence);
    void StoreMemRanges(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size);
    bool ValidateIdleDescriptorSet(VkDescriptorSet set, const char* func_str);
    void InitializeAndTrackMemory(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size, void** ppData);
    bool ValidatePipelineLocked(std::vector<std::unique_ptr<PIPELINE_STATE>> const& pPipelines, int pipelineIndex);
    bool ValidatePipelineUnlocked(std::vector<std::unique_ptr<PIPELINE_STATE>> const& pPipelines, int pipelineIndex);
    void FreeDescriptorSet(cvdescriptorset::DescriptorSet* descriptor_set);
    void DeletePools();
    bool ValidImageBufferQueue(CMD_BUFFER_STATE* cb_node, const VK_OBJECT* object, VkQueue queue, uint32_t count,
                               const uint32_t* indices);
    bool ValidateFenceForSubmit(FENCE_STATE* pFence);
    void AddMemObjInfo(void* object, const VkDeviceMemory mem, const VkMemoryAllocateInfo* pAllocateInfo);
    bool ValidateStatus(CMD_BUFFER_STATE* pNode, CBStatusFlags status_mask, VkFlags msg_flags, const char* fail_msg,
                        const char* msg_code);
    bool ValidateDrawStateFlags(CMD_BUFFER_STATE* pCB, const PIPELINE_STATE* pPipe, bool indexed, const char* msg_code);
    bool LogInvalidAttachmentMessage(const char* type1_string, const RENDER_PASS_STATE* rp1_state, const char* type2_string,
                                     const RENDER_PASS_STATE* rp2_state, uint32_t primary_attach, uint32_t secondary_attach,
                                     const char* msg, const char* caller, const char* error_code);
    bool ValidateStageMaskGsTsEnables(VkPipelineStageFlags stageMask, const char* caller, const char* geo_error_id,
                                      const char* tess_error_id, const char* mesh_error_id, const char* task_error_id);
    bool ValidateMapMemRange(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size);
    bool ValidatePushConstantRange(const uint32_t offset, const uint32_t size, const char* caller_name, uint32_t index);
    bool ValidateRenderPassDAG(RenderPassCreateVersion rp_version, const VkRenderPassCreateInfo2KHR* pCreateInfo,
                               RENDER_PASS_STATE* render_pass);
    bool ValidateAttachmentCompatibility(const char* type1_string, const RENDER_PASS_STATE* rp1_state, const char* type2_string,
                                         const RENDER_PASS_STATE* rp2_state, uint32_t primary_attach, uint32_t secondary_attach,
                                         const char* caller, const char* error_code);
    bool ValidateSubpassCompatibility(const char* type1_string, const RENDER_PASS_STATE* rp1_state, const char* type2_string,
                                      const RENDER_PASS_STATE* rp2_state, const int subpass, const char* caller,
                                      const char* error_code);
    bool ValidateRenderPassCompatibility(const char* type1_string, const RENDER_PASS_STATE* rp1_state, const char* type2_string,
                                         const RENDER_PASS_STATE* rp2_state, const char* caller, const char* error_code);
    void UpdateDrawState(CMD_BUFFER_STATE* cb_state, const VkPipelineBindPoint bind_point);
    bool ReportInvalidCommandBuffer(const CMD_BUFFER_STATE* cb_state, const char* call_source);
    void InitGpuValidation();
    bool ValidatePhysicalDeviceQueueFamily(const PHYSICAL_DEVICE_STATE* pd_state, uint32_t requested_queue_family,
                                           const char* err_code, const char* cmd_name, const char* queue_family_var_name);
    bool ValidateDeviceQueueCreateInfos(const PHYSICAL_DEVICE_STATE* pd_state, uint32_t info_count,
                                        const VkDeviceQueueCreateInfo* infos);

    bool ValidatePipelineVertexDivisors(std::vector<std::unique_ptr<PIPELINE_STATE>> const& pipe_state_vec, const uint32_t count,
                                        const VkGraphicsPipelineCreateInfo* pipe_cis);
    void AddFramebufferBinding(CMD_BUFFER_STATE* cb_state, FRAMEBUFFER_STATE* fb_state);
    bool ValidateImageBarrierImage(const char* funcName, CMD_BUFFER_STATE const* cb_state, VkFramebuffer framebuffer,
                                   uint32_t active_subpass, const safe_VkSubpassDescription2KHR& sub_desc, uint64_t rp_handle,
                                   uint32_t img_index, const VkImageMemoryBarrier& img_barrier);
    void RecordCmdBeginRenderPassState(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin,
                                       const VkSubpassContents contents);
    bool ValidateCmdBeginRenderPass(VkCommandBuffer commandBuffer, RenderPassCreateVersion rp_version,
                                    const VkRenderPassBeginInfo* pRenderPassBegin);
    bool ValidateDependencies(FRAMEBUFFER_STATE const* framebuffer, RENDER_PASS_STATE const* renderPass);
    bool ValidateBarriers(const char* funcName, CMD_BUFFER_STATE* cb_state, VkPipelineStageFlags src_stage_mask,
                          VkPipelineStageFlags dst_stage_mask, uint32_t memBarrierCount, const VkMemoryBarrier* pMemBarriers,
                          uint32_t bufferBarrierCount, const VkBufferMemoryBarrier* pBufferMemBarriers,
                          uint32_t imageMemBarrierCount, const VkImageMemoryBarrier* pImageMemBarriers);
    void RecordCmdPushDescriptorSetState(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
                                         uint32_t set, uint32_t descriptorWriteCount,
                                         const VkWriteDescriptorSet* pDescriptorWrites);
    void UpdateLastBoundDescriptorSets(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint pipeline_bind_point,
                                       const PIPELINE_LAYOUT_STATE* pipeline_layout, uint32_t first_set, uint32_t set_count,
                                       const std::vector<cvdescriptorset::DescriptorSet*> descriptor_sets,
                                       uint32_t dynamic_offset_count, const uint32_t* p_dynamic_offsets);
    bool ValidatePipelineBindPoint(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint bind_point, const char* func_name,
                                   const std::map<VkPipelineBindPoint, std::string>& bind_errors);
    bool ValidateMemoryIsMapped(const char* funcName, uint32_t memRangeCount, const VkMappedMemoryRange* pMemRanges);
    bool ValidateAndCopyNoncoherentMemoryToDriver(uint32_t mem_range_count, const VkMappedMemoryRange* mem_ranges);
    void CopyNoncoherentMemoryFromDriver(uint32_t mem_range_count, const VkMappedMemoryRange* mem_ranges);
    bool ValidateMappedMemoryRangeDeviceLimits(const char* func_name, uint32_t mem_range_count,
                                               const VkMappedMemoryRange* mem_ranges);
    BarrierOperationsType ComputeBarrierOperationsType(CMD_BUFFER_STATE* cb_state, uint32_t buffer_barrier_count,
                                                       const VkBufferMemoryBarrier* buffer_barriers, uint32_t image_barrier_count,
                                                       const VkImageMemoryBarrier* image_barriers);
    bool ValidateStageMasksAgainstQueueCapabilities(CMD_BUFFER_STATE const* cb_state, VkPipelineStageFlags source_stage_mask,
                                                    VkPipelineStageFlags dest_stage_mask, BarrierOperationsType barrier_op_type,
                                                    const char* function, const char* error_code);
    bool SetEventStageMask(VkQueue queue, VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask);
    bool ValidateRenderPassImageBarriers(const char* funcName, CMD_BUFFER_STATE* cb_state, uint32_t active_subpass,
                                         const safe_VkSubpassDescription2KHR& sub_desc, uint64_t rp_handle,
                                         const safe_VkSubpassDependency2KHR* dependencies,
                                         const std::vector<uint32_t>& self_dependencies, uint32_t image_mem_barrier_count,
                                         const VkImageMemoryBarrier* image_barriers);
    bool ValidateSecondaryCommandBufferState(CMD_BUFFER_STATE* pCB, CMD_BUFFER_STATE* pSubCB);
    bool ValidateFramebuffer(VkCommandBuffer primaryBuffer, const CMD_BUFFER_STATE* pCB, VkCommandBuffer secondaryBuffer,
                             const CMD_BUFFER_STATE* pSubCB, const char* caller);
    bool ValidateDescriptorUpdateTemplate(const char* func_name, const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo);
    bool ValidateCreateSamplerYcbcrConversion(const char* func_name, const VkSamplerYcbcrConversionCreateInfo* create_info);
    void RecordCreateSamplerYcbcrConversionState(const VkSamplerYcbcrConversionCreateInfo* create_info,
                                                 VkSamplerYcbcrConversion ycbcr_conversion);
    bool ValidateImportFence(VkFence fence, const char* caller_name);
    void RecordImportFenceState(VkFence fence, VkExternalFenceHandleTypeFlagBitsKHR handle_type, VkFenceImportFlagsKHR flags);
    void RecordGetExternalFenceState(VkFence fence, VkExternalFenceHandleTypeFlagBitsKHR handle_type);
    bool ValidateAcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence,
                                  uint32_t* pImageIndex, const char* func_name);
    void RecordAcquireNextImageState(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                                     VkFence fence, uint32_t* pImageIndex);
    bool VerifyRenderAreaBounds(const VkRenderPassBeginInfo* pRenderPassBegin);
    bool ValidatePrimaryCommandBuffer(const CMD_BUFFER_STATE* pCB, char const* cmd_name, const char* error_code);
    void RecordCmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents);
    bool ValidateCmdEndRenderPass(RenderPassCreateVersion rp_version, VkCommandBuffer commandBuffer);
    void RecordCmdEndRenderPassState(VkCommandBuffer commandBuffer);
    bool ValidateFramebufferCreateInfo(const VkFramebufferCreateInfo* pCreateInfo);
    bool MatchUsage(uint32_t count, const VkAttachmentReference2KHR* attachments, const VkFramebufferCreateInfo* fbci,
                    VkImageUsageFlagBits usage_flag, const char* error_code);
    bool ValidateBindImageMemory(VkImage image, VkDeviceMemory mem, VkDeviceSize memoryOffset, const char* api_name);
    void UpdateBindImageMemoryState(VkImage image, VkDeviceMemory mem, VkDeviceSize memoryOffset);
    void RecordGetPhysicalDeviceDisplayPlanePropertiesState(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount,
                                                            void* pProperties);
    bool ValidateGetPhysicalDeviceDisplayPlanePropertiesKHRQuery(VkPhysicalDevice physicalDevice, uint32_t planeIndex,
                                                                 const char* api_name);
    bool ValidateQuery(VkQueue queue, CMD_BUFFER_STATE* pCB, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount);
    bool IsQueryInvalid(QUEUE_STATE* queue_data, VkQueryPool queryPool, uint32_t queryIndex);
    bool ValidateImportSemaphore(VkSemaphore semaphore, const char* caller_name);
    void RecordImportSemaphoreState(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBitsKHR handle_type,
                                    VkSemaphoreImportFlagsKHR flags);
    void RecordGetExternalSemaphoreState(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBitsKHR handle_type);
    bool ValidateBeginQuery(const CMD_BUFFER_STATE* cb_state, const QueryObject& query_obj, VkFlags flags, CMD_TYPE cmd,
                            const char* cmd_name, const char* vuid_queue_flags, const char* vuid_queue_feedback,
                            const char* vuid_queue_occlusion, const char* vuid_precise, const char* vuid_query_count);
    void RecordBeginQuery(CMD_BUFFER_STATE* cb_state, const QueryObject& query_obj);
    bool ValidateCmdEndQuery(const CMD_BUFFER_STATE* cb_state, const QueryObject& query_obj, CMD_TYPE cmd, const char* cmd_name,
                             const char* vuid_queue_flags, const char* vuid_active_queries);
    void RecordCmdEndQuery(CMD_BUFFER_STATE* cb_state, const QueryObject& query_obj);

    bool SetQueryState(VkQueue queue, VkCommandBuffer commandBuffer, QueryObject object, bool value);
    bool ValidateCmdDrawType(VkCommandBuffer cmd_buffer, bool indexed, VkPipelineBindPoint bind_point, CMD_TYPE cmd_type,
                             const char* caller, VkQueueFlags queue_flags, const char* queue_flag_code,
                             const char* renderpass_msg_code, const char* pipebound_msg_code, const char* dynamic_state_msg_code);
    bool ValidateCmdNextSubpass(RenderPassCreateVersion rp_version, VkCommandBuffer commandBuffer);
    void RecordVulkanSurface(VkSurfaceKHR* pSurface);
    void PostRecordEnumeratePhysicalDeviceGroupsState(uint32_t* pPhysicalDeviceGroupCount,
                                                      VkPhysicalDeviceGroupPropertiesKHR* pPhysicalDeviceGroupProperties);
    void RecordCreateDescriptorUpdateTemplateState(const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo,
                                                   VkDescriptorUpdateTemplateKHR* pDescriptorUpdateTemplate);
    void RecordGetDeviceQueueState(uint32_t queue_family_index, VkQueue queue);
    bool AddAttachmentUse(RenderPassCreateVersion rp_version, uint32_t subpass, std::vector<uint8_t>& attachment_uses,
                          std::vector<VkImageLayout>& attachment_layouts, uint32_t attachment, uint8_t new_use,
                          VkImageLayout new_layout);
    bool CheckStageMaskQueueCompatibility(VkCommandBuffer command_buffer, VkPipelineStageFlags stage_mask, VkQueueFlags queue_flags,
                                          const char* function, const char* src_or_dest, const char* error_code);
    void RecordUpdateDescriptorSetWithTemplateState(VkDescriptorSet descriptorSet,
                                                    VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, const void* pData);

    void PostCallRecordEnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount,
                                                VkPhysicalDevice* pPhysicalDevices, VkResult result);

    // Prototypes for CoreChecks accessor functions
    const VkPhysicalDeviceMemoryProperties* GetPhysicalDeviceMemoryProperties();

    void RetireWorkOnQueue(QUEUE_STATE* pQueue, uint64_t seq);

    // Descriptor Set Validation Functions
    void PerformUpdateDescriptorSetsWithTemplateKHR(VkDescriptorSet descriptorSet, const TEMPLATE_STATE* template_state,
                                                    const void* pData);
    void UpdateAllocateDescriptorSetsData(const VkDescriptorSetAllocateInfo*, cvdescriptorset::AllocateDescriptorSetsData*);
    void PerformAllocateDescriptorSets(const VkDescriptorSetAllocateInfo*, const VkDescriptorSet*,
                                       const cvdescriptorset::AllocateDescriptorSetsData*);

    // Stuff from shader_validation
    void PreCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                         const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule, void* csm_state);
    void PostCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                          const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule, VkResult result,
                                          void* csm_state);
    bool ValidatePointListShaderState(const PIPELINE_STATE* pipeline, SHADER_MODULE_STATE const* src, spirv_inst_iter entrypoint,
                                      VkShaderStageFlagBits stage);
    bool ValidateShaderCapabilities(SHADER_MODULE_STATE const* src, VkShaderStageFlagBits stage, bool has_writable_descriptor);
    bool ValidateShaderStageInputOutputLimits(SHADER_MODULE_STATE const* src, VkPipelineShaderStageCreateInfo const* pStage,
                                              PIPELINE_STATE* pipeline);
    bool ValidateCooperativeMatrix(SHADER_MODULE_STATE const* src, VkPipelineShaderStageCreateInfo const* pStage,
                                   PIPELINE_STATE* pipeline);
    bool ValidateExecutionModes(SHADER_MODULE_STATE const* src, spirv_inst_iter entrypoint);

    // Gpu Validation Functions
    void GpuPreCallRecordCreateDevice(VkPhysicalDevice gpu, std::unique_ptr<safe_VkDeviceCreateInfo>& modified_create_info,
                                      VkPhysicalDeviceFeatures* supported_features);
    void GpuPostCallRecordCreateDevice(const CHECK_ENABLED* enables);
    void GpuPreCallRecordDestroyDevice();
    void GpuResetCommandBuffer(const VkCommandBuffer commandBuffer);
    bool GpuPreCallCreateShaderModule(const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                                      VkShaderModule* pShaderModule, uint32_t* unique_shader_id,
                                      VkShaderModuleCreateInfo* instrumented_create_info,
                                      std::vector<unsigned int>* instrumented_pgm);
    bool GpuPreCallCreatePipelineLayout(const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                                        VkPipelineLayout* pPipelineLayout, std::vector<VkDescriptorSetLayout>* new_layouts,
                                        VkPipelineLayoutCreateInfo* modified_create_info);
    void GpuPostCallCreatePipelineLayout(VkResult result);
    void GpuPreCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
    void GpuPostCallQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
    void GpuPreCallValidateCmdWaitEvents(VkPipelineStageFlags sourceStageMask);
    std::vector<safe_VkGraphicsPipelineCreateInfo> GpuPreCallRecordCreateGraphicsPipelines(
        VkPipelineCache pipelineCache, uint32_t count, const VkGraphicsPipelineCreateInfo* pCreateInfos,
        const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines, std::vector<std::unique_ptr<PIPELINE_STATE>>& pipe_state);
    void GpuPostCallRecordCreateGraphicsPipelines(const uint32_t count, const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                                  const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);
    void GpuPreCallRecordDestroyPipeline(const VkPipeline pipeline);
    void GpuAllocateValidationResources(const VkCommandBuffer cmd_buffer, VkPipelineBindPoint bind_point);
    void AnalyzeAndReportError(CMD_BUFFER_STATE* cb_node, VkQueue queue, uint32_t draw_index, uint32_t* const debug_output_buffer);
    void ProcessInstrumentationBuffer(VkQueue queue, CMD_BUFFER_STATE* cb_node);
    void UpdateInstrumentationBuffer(CMD_BUFFER_STATE* cb_node);
    void SubmitBarrier(VkQueue queue);
    bool GpuInstrumentShader(const VkShaderModuleCreateInfo* pCreateInfo, std::vector<unsigned int>& new_pgm,
                             uint32_t* unique_shader_id);
    VkResult GpuInitializeVma();
    void ReportSetupProblem(VkDebugReportObjectTypeEXT object_type, uint64_t object_handle, const char* const specific_message);

    bool ValidateIdleBuffer(VkBuffer buffer);
    bool ValidateUsageFlags(VkFlags actual, VkFlags desired, VkBool32 strict, uint64_t obj_handle, VulkanObjectType obj_type,
                            const char* msgCode, char const* func_name, char const* usage_str);
    bool ValidateImageSubresourceRange(const uint32_t image_mip_count, const uint32_t image_layer_count,
                                       const VkImageSubresourceRange& subresourceRange, const char* cmd_name,
                                       const char* param_name, const char* image_layer_count_var_name, const uint64_t image_handle,
                                       SubresourceRangeErrorCodes errorCodes);
    bool ValidateRenderPassLayoutAgainstFramebufferImageUsage(RenderPassCreateVersion rp_version, VkImageLayout layout,
                                                              VkImage image, VkImageView image_view, VkFramebuffer framebuffer,
                                                              VkRenderPass renderpass, uint32_t attachment_index,
                                                              const char* variable_name);

    void PreCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                              const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                              const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines, void* cgpl_state);
    void PostCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                               const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                               const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines, VkResult result,
                                               void* cgpl_state);
    void PostCallRecordCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                              const VkComputePipelineCreateInfo* pCreateInfos,
                                              const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines, VkResult result,
                                              void* pipe_state);
    void PreCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout,
                                           void* cpl_state);
    void PostCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo,
                                            const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout,
                                            VkResult result);
    bool PreCallValidateAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                               VkDescriptorSet* pDescriptorSets, void* ads_state);
    void PostCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                              VkDescriptorSet* pDescriptorSets, VkResult result, void* ads_state);
    void PostCallRecordCreateRayTracingPipelinesNV(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                   const VkRayTracingPipelineCreateInfoNV* pCreateInfos,
                                                   const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines, VkResult result,
                                                   void* pipe_state);
    void PostCallRecordCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                                      VkInstance* pInstance, VkResult result);
    void PreCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo* pCreateInfo,
                                   const VkAllocationCallbacks* pAllocator, VkDevice* pDevice,
                                   std::unique_ptr<safe_VkDeviceCreateInfo>& modified_create_info);
    void PostCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo* pCreateInfo,
                                    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice, VkResult result);
    void PostCallRecordGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue);
    void PostCallRecordGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* pQueueInfo, VkQueue* pQueue);
    void PreCallRecordDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator);
    void PreCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
    void PostCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence,
                                   VkResult result);
    void PostCallRecordQueueWaitIdle(VkQueue queue, VkResult result);
    void PostCallRecordDeviceWaitIdle(VkDevice device, VkResult result);
    void PreCallRecordDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);
    void PreCallRecordDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator);
    void PreCallRecordDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                 const VkAllocationCallbacks* pAllocator);
    void PreCallRecordDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                            const VkAllocationCallbacks* pAllocator);
    void PreCallRecordFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                         const VkCommandBuffer* pCommandBuffers);
    void PostCallRecordCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout,
                                                 VkResult result);
    void PreCallRecordUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                           const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount,
                                           const VkCopyDescriptorSet* pDescriptorCopies);
    void PostCallRecordAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pCreateInfo,
                                              VkCommandBuffer* pCommandBuffer, VkResult result);
    void PostCallRecordResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags, VkResult result);
    void PreCallRecordCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline);
    void PreCallRecordCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                            VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
                                            const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount,
                                            const uint32_t* pDynamicOffsets);
    void PreCallRecordCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                              VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                              const VkWriteDescriptorSet* pDescriptorWrites);
    void PreCallRecordCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents,
                                    VkPipelineStageFlags sourceStageMask, VkPipelineStageFlags dstStageMask,
                                    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
                                    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
                                    uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers);
    void PreCallRecordCmdBeginDebugUtilsLabelEXT(VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT* pLabelInfo);
    void PostCallRecordCmdEndDebugUtilsLabelEXT(VkCommandBuffer commandBuffer);
    void PreCallRecordCmdInsertDebugUtilsLabelEXT(VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT* pLabelInfo);
    void PreCallRecordGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                  VkPhysicalDeviceProperties* pPhysicalDeviceProperties);

    void InvalidateCommandBuffers(std::unordered_set<CMD_BUFFER_STATE*> const& cb_nodes, VK_OBJECT obj);

    // From DrawDispatch, temporarily
    void PreCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                              uint32_t firstInstance);

    void PreCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                     uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

    void PreCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                      uint32_t stride);

    void PreCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                             uint32_t stride);

    void PreCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);

    void PreCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);

};  // Class CoreChecks
