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
#include <SPIRV/spirv.hpp>
#include <spirv_tools_commit_id.h>
#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <deque>

// A forward iterator over spirv instructions. Provides easy access to len, opcode, and content words
// without the caller needing to care too much about the physical SPIRV module layout.
struct spirv_inst_iter {
    std::vector<uint32_t>::const_iterator zero;
    std::vector<uint32_t>::const_iterator it;

    uint32_t len() {
        auto result = *it >> 16;
        assert(result > 0);
        return result;
    }

    uint32_t opcode() { return *it & 0x0ffffu; }

    uint32_t const& word(unsigned n) {
        assert(n < len());
        return it[n];
    }

    uint32_t offset() { return (uint32_t)(it - zero); }

    spirv_inst_iter() {}

    spirv_inst_iter(std::vector<uint32_t>::const_iterator zero, std::vector<uint32_t>::const_iterator it) : zero(zero), it(it) {}

    bool operator==(spirv_inst_iter const& other) { return it == other.it; }

    bool operator!=(spirv_inst_iter const& other) { return it != other.it; }

    spirv_inst_iter operator++(int) {  // x++
        spirv_inst_iter ii = *this;
        it += len();
        return ii;
    }

    spirv_inst_iter operator++() {  // ++x;
        it += len();
        return *this;
    }

    // The iterator and the value are the same thing.
    spirv_inst_iter& operator*() { return *this; }
    spirv_inst_iter const& operator*() const { return *this; }
};

struct SHADER_MODULE_STATE {
    // The spirv image itself
    std::vector<uint32_t> words;
    bool has_valid_spirv;
    VkShaderModule vk_shader_module;
    uint32_t gpu_validation_shader_id;

    SHADER_MODULE_STATE(VkShaderModuleCreateInfo const* pCreateInfo, VkShaderModule shaderModule, uint32_t unique_shader_id)
        : words((uint32_t*)pCreateInfo->pCode, (uint32_t*)pCreateInfo->pCode + pCreateInfo->codeSize / sizeof(uint32_t)),
          has_valid_spirv(true),
          vk_shader_module(shaderModule),
          gpu_validation_shader_id(unique_shader_id) {}

    SHADER_MODULE_STATE() : has_valid_spirv(false), vk_shader_module(VK_NULL_HANDLE) {}

    // Expose begin() / end() to enable range-based for
    spirv_inst_iter begin() const { return spirv_inst_iter(words.begin(), words.begin() + 5); }  // First insn
    spirv_inst_iter end() const { return spirv_inst_iter(words.begin(), words.end()); }          // Just past last insn
    // Given an offset into the module, produce an iterator there.
    spirv_inst_iter at(unsigned offset) const { return spirv_inst_iter(words.begin(), words.begin() + offset); }
};

struct PHYSICAL_DEVICE_STATE {
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    uint32_t queue_family_count = 0;
    std::vector<VkQueueFamilyProperties> queue_family_properties;
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

using std::unordered_map;
struct GpuValidationState;

class CoreChecks : public ValidationObject {
   public:
    unordered_map<VkPipeline, std::unique_ptr<PIPELINE_STATE>> pipelineMap;
    unordered_map<VkShaderModule, std::unique_ptr<SHADER_MODULE_STATE>> shaderModuleMap;
    unordered_map<VkDescriptorUpdateTemplateKHR, std::unique_ptr<TEMPLATE_STATE>> desc_template_map;
    unordered_map<VkDescriptorPool, std::unique_ptr<DESCRIPTOR_POOL_STATE>> descriptorPoolMap;
    unordered_map<VkDescriptorSet, std::unique_ptr<cvdescriptorset::DescriptorSet>> setMap;
    unordered_map<VkCommandBuffer, std::unique_ptr<CMD_BUFFER_STATE>> commandBufferMap;
    unordered_map<VkCommandPool, std::unique_ptr<COMMAND_POOL_STATE>> commandPoolMap;
    unordered_map<VkPipelineLayout, std::unique_ptr<PIPELINE_LAYOUT_STATE>> pipelineLayoutMap;
    unordered_map<VkQueue, uint32_t> queueMap;
    unordered_map<VkRenderPass, std::shared_ptr<RENDER_PASS_STATE>> renderPassMap;
    unordered_map<VkDescriptorSetLayout, std::shared_ptr<cvdescriptorset::DescriptorSetLayout>> descriptorSetLayoutMap;

    // Used for instance versions of this object
    unordered_map<VkPhysicalDevice, PHYSICAL_DEVICE_STATE> physical_device_map;
    // Link to the device's physical-device data
    PHYSICAL_DEVICE_STATE* physical_device_state;

    // Link for derived device objects back to their parent instance object
    CoreChecks* instance_state;

    // Device specific data
    VkPhysicalDeviceProperties phys_dev_props = {};
    std::unique_ptr<GpuValidationState> gpu_validation_state;

    // Class Declarations for helper functions
    cvdescriptorset::DescriptorSet* GetSetNode(VkDescriptorSet);
    DESCRIPTOR_POOL_STATE* GetDescriptorPoolState(const VkDescriptorPool);
    CMD_BUFFER_STATE* GetCBState(const VkCommandBuffer cb);
    PIPELINE_STATE* GetPipelineState(VkPipeline pipeline);
    RENDER_PASS_STATE* GetRenderPassState(VkRenderPass renderpass);
    std::shared_ptr<RENDER_PASS_STATE> GetRenderPassStateSharedPtr(VkRenderPass renderpass);
    COMMAND_POOL_STATE* GetCommandPoolState(VkCommandPool pool);
    SHADER_MODULE_STATE const* GetShaderModuleState(VkShaderModule module);
    uint32_t GetQueueState(VkQueue queue);
    PHYSICAL_DEVICE_STATE* GetPhysicalDeviceState(VkPhysicalDevice phys);
    PHYSICAL_DEVICE_STATE* GetPhysicalDeviceState();

    void PreCallRecordDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator);
    void RecordCreateRenderPassState(RenderPassCreateVersion rp_version, std::shared_ptr<RENDER_PASS_STATE>& render_pass,
                                     VkRenderPass* pRenderPass);
    void PostCallRecordCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo,
                                        const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass, VkResult result);
    void PostCallRecordCreateRenderPass2KHR(VkDevice device, const VkRenderPassCreateInfo2KHR* pCreateInfo,
                                            const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass, VkResult result);

    void UpdateStateCmdDrawDispatchType(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint bind_point);
    void UpdateStateCmdDrawType(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint bind_point);
    void PostCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
    void PostCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
    void PostCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                               uint32_t firstInstance);
    void PostCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                      uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);
    void PostCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                       uint32_t stride);
    void PostCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                              uint32_t stride);
    void PreCallRecordCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                              VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                              uint32_t stride);
    void PreCallRecordCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                     VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                     uint32_t stride);
    void PreCallRecordCmdDrawMeshTasksNV(VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask);
    void PreCallRecordCmdDrawMeshTasksIndirectNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                 uint32_t drawCount, uint32_t stride);
    void PreCallRecordCmdDrawMeshTasksIndirectCountNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                      VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                      uint32_t stride);
    bool PreCallValidateCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                                const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines,
                                                void* cgpl_state_data);
    void ResetCommandBufferState(const VkCommandBuffer cb);
    PIPELINE_LAYOUT_STATE const* GetPipelineLayout(VkPipelineLayout pipeLayout);
    void FreeCommandBufferStates(COMMAND_POOL_STATE* pool_state, const uint32_t command_buffer_count,
                                 const VkCommandBuffer* command_buffers);
    void FreeDescriptorSet(cvdescriptorset::DescriptorSet* descriptor_set);
    void DeletePools();
    void UpdateDrawState(CMD_BUFFER_STATE* cb_state, const VkPipelineBindPoint bind_point);
    void InitGpuValidation();
    void RecordCmdPushDescriptorSetState(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
                                         uint32_t set, uint32_t descriptorWriteCount,
                                         const VkWriteDescriptorSet* pDescriptorWrites);
    void UpdateLastBoundDescriptorSets(CMD_BUFFER_STATE* cb_state, VkPipelineBindPoint pipeline_bind_point,
                                       const PIPELINE_LAYOUT_STATE* pipeline_layout, uint32_t first_set, uint32_t set_count,
                                       const std::vector<cvdescriptorset::DescriptorSet*> descriptor_sets,
                                       uint32_t dynamic_offset_count, const uint32_t* p_dynamic_offsets);

    void PostCallRecordEnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount,
                                                VkPhysicalDevice* pPhysicalDevices, VkResult result);

    // Descriptor Set Validation Functions
    void UpdateAllocateDescriptorSetsData(const VkDescriptorSetAllocateInfo*, cvdescriptorset::AllocateDescriptorSetsData*);
    void PerformAllocateDescriptorSets(const VkDescriptorSetAllocateInfo*, const VkDescriptorSet*,
                                       const cvdescriptorset::AllocateDescriptorSetsData*);

    // Stuff from shader_validation
    void PreCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                         const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule, void* csm_state);
    void PostCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                          const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule, VkResult result,
                                          void* csm_state);

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
    void PreCallRecordDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator);
    void PreCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
    void PostCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence,
                                   VkResult result);
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
    void PreCallRecordGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                  VkPhysicalDeviceProperties* pPhysicalDeviceProperties);
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
